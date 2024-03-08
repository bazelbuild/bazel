// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.vfs;

import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.base.Throwables.throwIfUnchecked;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ForwardingListenableFuture;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * Implementation of a subset of UNIX-style file globbing, expanding "*" and "?" as wildcards, but
 * not [a-z] ranges.
 *
 * <p><code>**</code> gets special treatment in include patterns. If it is used as a complete path
 * segment it matches the filenames in subdirectories recursively.
 *
 * <p>Importantly, note that the glob matches are in an unspecified order.
 */
public final class UnixGlob {
  private static final UnixGlobPathDiscriminator DEFAULT_DISCRIMINATOR =
      new UnixGlobPathDiscriminator() {};

  private UnixGlob() {}

  /** Indicates an invalid glob pattern. */
  public static final class BadPattern extends Exception {
    private BadPattern(String message) {
      super(message);
    }
  }

  private static List<Path> globInternal(
      Path base,
      Collection<String> patterns,
      UnixGlobPathDiscriminator pathDiscriminator,
      SyscallCache syscalls,
      Executor executor)
      throws IOException, InterruptedException, BadPattern {
    GlobVisitor visitor = new GlobVisitor(executor);
    return visitor.glob(base, patterns, pathDiscriminator, syscalls);
  }

  private static List<Path> globInternalUninterruptible(
      Path base,
      Collection<String> patterns,
      UnixGlobPathDiscriminator pathDiscriminator,
      SyscallCache syscalls,
      Executor executor)
      throws IOException, BadPattern {
    GlobVisitor visitor = new GlobVisitor(executor);
    return visitor.globUninterruptible(base, patterns, pathDiscriminator, syscalls);
  }

  private static long globInternalAndReturnNumGlobTasksForTesting(
      Path base,
      Collection<String> patterns,
      UnixGlobPathDiscriminator pathDiscriminator,
      SyscallCache syscalls,
      Executor executor)
      throws IOException, InterruptedException, BadPattern {
    GlobVisitor visitor = new GlobVisitor(executor);
    visitor.glob(base, patterns, pathDiscriminator, syscalls);
    return visitor.getNumGlobTasksForTesting();
  }

  private static Future<List<Path>> globAsyncInternal(
      Path base,
      Collection<String> patterns,
      UnixGlobPathDiscriminator pathDiscriminator,
      SyscallCache syscalls,
      Executor executor)
      throws BadPattern {
    Preconditions.checkNotNull(executor, "%s %s", base, patterns);
    return new GlobVisitor(executor).globAsync(base, patterns, pathDiscriminator, syscalls);
  }

  /**
   * Checks that each pattern is valid, splits it into segments and checks that each segment
   * contains only valid wildcards.
   *
   * @throws BadPattern on encountering a malformed pattern.
   * @return list of segment arrays
   */
  private static List<String[]> checkAndSplitPatterns(Collection<String> patterns)
      throws BadPattern {
    List<String[]> list = Lists.newArrayListWithCapacity(patterns.size());
    for (String pattern : patterns) {
      String error = checkPatternForError(pattern);
      if (error != null) {
        throw new BadPattern(error + " (in glob pattern '" + pattern + "')");
      }
      Iterable<String> segments = Splitter.on('/').split(pattern);
      list.add(Iterables.toArray(segments, String.class));
    }
    return list;
  }

  /**
   * @return whether or not {@code pattern} contains illegal characters
   */
  @Nullable
  public static String checkPatternForError(String pattern) {
    if (pattern.isEmpty()) {
      return "pattern cannot be empty";
    }
    if (pattern.charAt(0) == '/') {
      return "pattern cannot be absolute";
    }
    Iterable<String> segments = Splitter.on('/').split(pattern);
    for (String segment : segments) {
      if (segment.isEmpty()) {
        return "empty segment not permitted";
      }
      if (segment.equals(".") || segment.equals("..")) {
        return "segment '" + segment + "' not permitted";
      }
      if (segment.contains("**") && !segment.equals("**")) {
        return "recursive wildcard must be its own segment";
      }
    }
    return null;
  }

  /** Calls {@link #matches(String, String, Map) matches(pattern, str, null)} */
  public static boolean matches(String pattern, String str) {
    return matches(pattern, str, null);
  }

  /**
   * Returns whether {@code str} matches the glob pattern {@code pattern}. This method may use the
   * {@code patternCache} to speed up the matching process.
   *
   * @param pattern a glob pattern
   * @param str the string to match
   * @param patternCache a cache from patterns to compiled Pattern objects, or {@code null} to skip
   *     caching
   */
  public static boolean matches(String pattern, String str, Map<String, Pattern> patternCache) {
    if (pattern.length() == 0 || str.length() == 0) {
      return false;
    }

    // Common case: **
    if (pattern.equals("**")) {
      return true;
    }

    // Common case: *
    if (pattern.equals("*")) {
      return true;
    }

    // If a filename starts with '.', this char must be matched explicitly.
    if (str.charAt(0) == '.' && pattern.charAt(0) != '.') {
      return false;
    }

    // Common case: *.xyz
    if (pattern.charAt(0) == '*' && pattern.lastIndexOf('*') == 0) {
      return str.endsWith(pattern.substring(1));
    }
    // Common case: xyz*
    int lastIndex = pattern.length() - 1;
    // The first clause of this if statement is unnecessary, but is an
    // optimization--charAt runs faster than indexOf.
    if (pattern.charAt(lastIndex) == '*' && pattern.indexOf('*') == lastIndex) {
      return str.startsWith(pattern.substring(0, lastIndex));
    }

    Pattern regex =
        patternCache == null
            ? makePatternFromWildcard(pattern)
            : patternCache.computeIfAbsent(pattern, p -> makePatternFromWildcard(p));
    return regex.matcher(str).matches();
  }

  /**
   * Returns a regular expression implementing a matcher for "pattern", in which
   * "*" and "?" are wildcards.
   *
   * <p>e.g. "foo*bar?.java" -> "foo.*bar.\\.java"
   */
  private static Pattern makePatternFromWildcard(String pattern) {
    StringBuilder regexp = new StringBuilder();
    for (int i = 0, len = pattern.length(); i < len; i++) {
      char c = pattern.charAt(i);
      switch (c) {
        case '*':
          int toIncrement = 0;
          if (len > i + 1 && pattern.charAt(i + 1) == '*') {
            // The pattern '**' is interpreted to match 0 or more directory separators, not 1 or
            // more. We skip the next * and then find a trailing/leading '/' and get rid of it.
            toIncrement = 1;
            if (len > i + 2 && pattern.charAt(i + 2) == '/') {
              // We have '**/' -- skip the '/'.
              toIncrement = 2;
            } else if (len == i + 2 && i > 0 && pattern.charAt(i - 1) == '/') {
              // We have '/**' -- remove the '/'.
              regexp.delete(regexp.length() - 1, regexp.length());
            }
          }
          regexp.append(".*");
          i += toIncrement;
          break;
        case '?':
          regexp.append('.');
          break;
        case '^':
        case '$':
        case '|':
        case '+':
        case '{':
        case '}':
        case '[':
        case ']':
        case '\\':
        case '.':
          // escape the regexp special characters that are allowed in wildcards
          regexp.append('\\');
          regexp.append(c);
          break;
        case '(':
        case ')':
          // The historical undocumented behavior of this function was to add '(' and ')' to the
          // regexp pattern string unescaped. That could have 2 effects: a no-op (if the parentheses
          // were properly paired) or a PatternSyntaxException leading to a Bazel crash (if the
          // parentheses were unpaired). The behavior was silly, but changing it will break existing
          // BUILD files which e.g. call `glob(["(*.foo)"])`. To keep such BUILD files working while
          // avoiding crashes, treat '(' and ')' as a safe no-op when compiling to regexp.
          // TODO(b/154003471): change this behavior and start treating '(' and ')' as literal
          // characters to match in a glob pattern. This change will require an incompatible flag.
          break;
        default:
          regexp.append(c);
          break;
      }
    }
    return Pattern.compile(regexp.toString());
  }

  /**
   * Builder class for UnixGlob.
   *
   *
   */
  public static class Builder {
    private final Path base;
    private final List<String> patterns;
    private final SyscallCache syscallCache;
    private UnixGlobPathDiscriminator pathDiscriminator = DEFAULT_DISCRIMINATOR;
    private Executor executor;

    /** Creates a glob builder with the given base path. */
    public Builder(Path base, SyscallCache syscallCache) {
      this.base = base;
      this.syscallCache = syscallCache;
      this.patterns = Lists.newArrayList();
    }

    /**
     * Adds a pattern to include to the glob builder.
     *
     * <p>For a description of the syntax of the patterns, see {@link UnixGlob}.
     */
    @CanIgnoreReturnValue
    public Builder addPattern(String pattern) {
      this.patterns.add(pattern);
      return this;
    }

    /**
     * Adds a pattern to include to the glob builder.
     *
     * <p>For a description of the syntax of the patterns, see {@link UnixGlob}.
     */
    @CanIgnoreReturnValue
    public Builder addPatterns(String... patterns) {
      Collections.addAll(this.patterns, patterns);
      return this;
    }

    /**
     * Adds a pattern to include to the glob builder.
     *
     * <p>For a description of the syntax of the patterns, see {@link UnixGlob}.
     */
    @CanIgnoreReturnValue
    public Builder addPatterns(Collection<String> patterns) {
      this.patterns.addAll(patterns);
      return this;
    }

    /**
     * Sets the executor to use for parallel glob evaluation. If unset, evaluation is done
     * in-thread.
     */
    @CanIgnoreReturnValue
    public Builder setExecutor(Executor pool) {
      this.executor = pool;
      return this;
    }

    /**
     * Sets the UnixGlobPathDiscriminator which determines how to handle Path entries encountered
     * during glob traversal. The interface determines if Paths should be added to the {@code
     * List<Path>} results and whether to traverse a given directory during recursion.
     *
     * <p>The UnixGlobPathDiscriminator should only be called with Paths that have been resolved to
     * a regular file or regular directory, it will not properly handle symlinks or special files.
     *
     * <p>This is used for handling the previous use case of 'excludeDirectories' where we wish to
     * exclude files from the glob and decide which directories to traverse, like skipping sub-dirs
     * containing BUILD files.
     */
    @CanIgnoreReturnValue
    public Builder setPathDiscriminator(UnixGlobPathDiscriminator pathDiscriminator) {
      this.pathDiscriminator = pathDiscriminator;
      return this;
    }

    /** Executes the glob. */
    public List<Path> glob() throws IOException, BadPattern {
      return globInternalUninterruptible(base, patterns, pathDiscriminator, syscallCache, executor);
    }

    /**
     * Executes the glob and returns the result.
     *
     * @throws InterruptedException if the thread is interrupted.
     */
    public List<Path> globInterruptible() throws IOException, InterruptedException, BadPattern {
      return globInternal(base, patterns, pathDiscriminator, syscallCache, executor);
    }

    @VisibleForTesting
    public long globInterruptibleAndReturnNumGlobTasksForTesting()
        throws IOException, InterruptedException, BadPattern {
      return globInternalAndReturnNumGlobTasksForTesting(
          base, patterns, pathDiscriminator, syscallCache, executor);
    }

    /**
     * Executes the glob asynchronously. {@link #setExecutor} must have been called already with a
     * non-null argument.
     */
    public Future<List<Path>> globAsync() throws BadPattern {
      return globAsyncInternal(base, patterns, pathDiscriminator, syscallCache, executor);
    }
  }

  /**
   * Adapts the result of the glob visitation as a Future.
   */
  private static class GlobFuture extends ForwardingListenableFuture<List<Path>> {
    private final GlobVisitor visitor;
    private final SettableFuture<List<Path>> delegate = SettableFuture.create();

    public GlobFuture(GlobVisitor visitor) {
      this.visitor = visitor;
    }

    @Override
    protected ListenableFuture<List<Path>> delegate() {
      return delegate;
    }

    public void setException(Throwable throwable) {
      delegate.setException(throwable);
    }

    public void set(List<Path> paths) {
      delegate.set(paths);
    }

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
      // Best-effort interrupt of the in-flight visitation.
      visitor.cancel();
      return true;
    }

    public void markCanceled() {
      super.cancel(true);
    }
  }

  /**
   * GlobVisitor executes a glob using parallelism, which is useful when
   * the glob() requires many readdir() calls on high latency filesystems.
   */
  private static final class GlobVisitor {
    // These collections are used across workers and must therefore be thread-safe.
    private final Collection<Path> results = Sets.newConcurrentHashSet();
    private final ConcurrentHashMap<String, Pattern> cache = new ConcurrentHashMap<>();

    private final GlobFuture result;
    private final Executor executor;
    private final AtomicLong totalOps = new AtomicLong(0);
    private final AtomicLong pendingOps = new AtomicLong(0);
    private final AtomicReference<IOException> ioException = new AtomicReference<>();
    private final AtomicReference<RuntimeException> runtimeException = new AtomicReference<>();
    private final AtomicReference<Error> error = new AtomicReference<>();
    private volatile boolean canceled = false;

    GlobVisitor(Executor executor) {
      this.executor = executor;
      this.result = new GlobFuture(this);
    }

    /**
     * Performs wildcard globbing: returns the list of filenames that match any of {@code patterns}
     * relative to {@code base}. Directories are traversed if and only if they return true from
     * {@code pathDiscriminator.shouldTraverseDirectory}. The predicate is also called for the root
     * of the traversal. {@code pathDiscriminator.shouldIncludePathInResult} is called to determine
     * if a directory result should be included in the output. The The order of the returned list is
     * unspecified.
     *
     * <p>Patterns may include "*" and "?", but not "[a-z]".
     *
     * <p><code>**</code> gets special treatment in include patterns. If it is used as a complete
     * path segment it matches the filenames in subdirectories recursively.
     *
     * @throws IllegalArgumentException if any glob pattern {@linkplain
     *     #checkPatternForError(String) contains errors} or if any include pattern segment contains
     *     <code>**</code> but not equal to it.
     */
    List<Path> glob(
        Path base,
        Collection<String> patterns,
        UnixGlobPathDiscriminator pathDiscriminator,
        SyscallCache syscalls)
        throws IOException, InterruptedException, BadPattern {
      try {
        return globAsync(base, patterns, pathDiscriminator, syscalls).get();
      } catch (ExecutionException e) {
        Throwable cause = e.getCause();
        throwIfInstanceOf(cause, IOException.class);
        throwIfUnchecked(cause);
        throw new RuntimeException(e);
      }
    }

    List<Path> globUninterruptible(
        Path base,
        Collection<String> patterns,
        UnixGlobPathDiscriminator pathDiscriminator,
        SyscallCache syscalls)
        throws IOException, BadPattern {
      try {
        return Uninterruptibles.getUninterruptibly(
            globAsync(base, patterns, pathDiscriminator, syscalls));
      } catch (ExecutionException e) {
        Throwable cause = e.getCause();
        throwIfInstanceOf(cause, IOException.class);
        throwIfInstanceOf(cause, BadPattern.class);
        throwIfUnchecked(cause);
        throw new RuntimeException(e);
      }
    }

    private static boolean isRecursivePattern(String pattern) {
      return "**".equals(pattern);
    }

    /**
     * Same as {@link #glob}, except does so asynchronously and returns a {@link Future} for the
     * result.
     */
    Future<List<Path>> globAsync(
        Path base,
        Collection<String> patterns,
        UnixGlobPathDiscriminator pathDiscriminator,
        SyscallCache syscalls)
        throws BadPattern {
      FileStatus baseStat;
      try {
        baseStat = syscalls.statIfFound(base, Symlinks.FOLLOW);
      } catch (IOException e) {
        return Futures.immediateFailedFuture(e);
      }
      if (baseStat == null || patterns.isEmpty()) {
        return Futures.immediateFuture(Collections.<Path>emptyList());
      }

      // TODO(adonovan): validate pattern unconditionally, before I/O (potentially breaking change).
      List<String[]> splitPatterns = checkAndSplitPatterns(patterns);

      // We do a dumb loop, even though it will likely duplicate logical work (note that the
      // physical filesystem operations are cached). In order to optimize, we would need to keep
      // track of which patterns shared sub-patterns and which did not (for example consider the
      // glob [*/*.java, sub/*.java, */*.txt]).
      pendingOps.incrementAndGet();
      try {
        for (String[] splitPattern : splitPatterns) {
          int numRecursivePatterns = 0;
          for (String pattern : splitPattern) {
            if (isRecursivePattern(pattern)) {
              ++numRecursivePatterns;
            }
          }
          GlobTaskContext context =
              numRecursivePatterns > 1
                  ? new RecursiveGlobTaskContext(splitPattern, pathDiscriminator, syscalls)
                  : new GlobTaskContext(splitPattern, pathDiscriminator, syscalls);
          context.queueGlob(base, baseStat.isDirectory(), 0);
        }
      } finally {
        decrementAndCheckDone();
      }

      return result;
    }

    @Nullable
    private Throwable getMostSeriousThrowableSoFar() {
      if (error.get() != null) {
        return error.get();
      }
      if (runtimeException.get() != null) {
        return runtimeException.get();
      }
      if (ioException.get() != null) {
        return ioException.get();
      }
      return null;
    }

    /** Should only be called by link {@link GlobTaskContext}. */
    private void queueGlob(
        final Path base, final boolean baseIsDir, final int idx, final GlobTaskContext context) {
      enqueue(
          new Runnable() {
            @Override
            public void run() {
              try (SilentCloseable c =
                  Profiler.instance().profile(ProfilerTask.VFS_GLOB, base.getPathString())) {
                reallyGlob(base, baseIsDir, idx, context);
              } catch (IOException e) {
                ioException.set(e);
              } catch (RuntimeException e) {
                runtimeException.set(e);
              } catch (Error e) {
                error.set(e);
              }
            }

            @Override
            public String toString() {
              return String.format(
                  "%s glob(include=[%s])",
                  base.getPathString(),
                  "\"" + Joiner.on("\", \"").join(context.patternParts) + "\"");
            }
          });
    }

    /** Should only be called by link {@link GlobTaskContext}. */
    private void queueTask(Runnable runnable) {
      enqueue(runnable);
    }

    void enqueue(final Runnable r) {
      totalOps.incrementAndGet();
      pendingOps.incrementAndGet();

      Runnable wrapped =
          () -> {
            try {
              if (!canceled && getMostSeriousThrowableSoFar() == null) {
                r.run();
              }
            } finally {
              decrementAndCheckDone();
            }
          };

      if (executor == null) {
        wrapped.run();
      } else {
        executor.execute(wrapped);
      }
    }

    private long getNumGlobTasksForTesting() {
      return totalOps.get();
    }

    void cancel() {
      this.canceled = true;
    }

    private void decrementAndCheckDone() {
      if (pendingOps.decrementAndGet() == 0) {
        // We get to 0 iff we are done all the relevant work. This is because we always increment
        // the pending ops count as we're enqueuing, and don't decrement until the task is complete
        // (which includes accounting for any additional tasks that one enqueues).

        Throwable mostSeriousThrowable = getMostSeriousThrowableSoFar();
        if (canceled) {
          result.markCanceled();
        } else if (mostSeriousThrowable != null) {
          result.setException(mostSeriousThrowable);
        } else {
          result.set(ImmutableList.copyOf(results));
        }
      }
    }

    /** A context for evaluating all the subtasks of a single top-level glob task. */
    private class GlobTaskContext {
      private final String[] patternParts;
      private final UnixGlobPathDiscriminator pathDiscriminator;
      private final SyscallCache syscalls;

      GlobTaskContext(
          String[] patternParts,
          UnixGlobPathDiscriminator pathDiscriminator,
          SyscallCache syscalls) {
        this.patternParts = patternParts;
        this.pathDiscriminator = pathDiscriminator;
        this.syscalls = syscalls;
      }

      protected void queueGlob(Path base, boolean baseIsDir, int patternIdx) {
        GlobVisitor.this.queueGlob(base, baseIsDir, patternIdx, this);
      }

      protected void queueTask(Runnable runnable) {
        GlobVisitor.this.queueTask(runnable);
      }
    }

    /**
     * A special implementation of {@link GlobTaskContext} that dedupes glob subtasks. Our naive
     * implementation of recursive patterns means there are multiple ways to enqueue the same
     * logical subtask.
     */
    private class RecursiveGlobTaskContext extends GlobTaskContext {

      private class GlobTask {
        private final Path base;
        private final int patternIdx;

        private GlobTask(Path base, int patternIdx) {
          this.base = base;
          this.patternIdx = patternIdx;
        }

        @Override
        public boolean equals(Object obj) {
          if (!(obj instanceof GlobTask)) {
            return false;
          }
          GlobTask other = (GlobTask) obj;
          return base.equals(other.base) && patternIdx == other.patternIdx;
        }

        @Override
        public int hashCode() {
          return Objects.hash(base, patternIdx);
        }
      }

      private final Set<GlobTask> visitedGlobSubTasks = Sets.newConcurrentHashSet();

      private RecursiveGlobTaskContext(
          String[] patternParts,
          UnixGlobPathDiscriminator pathDiscriminator,
          SyscallCache syscalls) {
        super(patternParts, pathDiscriminator, syscalls);
      }

      @Override
      protected void queueGlob(Path base, boolean baseIsDir, int patternIdx) {
        if (visitedGlobSubTasks.add(new GlobTask(base, patternIdx))) {
          // This is a unique glob task. For example of how duplicates can arise, consider:
          //   glob(['**/a/**/foo.txt'])
          // with the only file being
          //   a/a/foo.txt
          //
          // there are multiple ways to reach a/a/foo.txt: one route starts by recursively globbing
          // 'a/**/foo.txt' in the base directory of the package, and another route starts by
          // recursively globbing '**/a/**/foo.txt' in subdirectory 'a'.
          super.queueGlob(base, baseIsDir, patternIdx);
        }
      }
    }

    /**
     * Expressed in Haskell:
     *
     * <pre>
     *  reallyGlob base []     = { base }
     *  reallyGlob base [x:xs] = union { reallyGlob(f, xs) | f results "base/x" }
     * </pre>
     */
    private void reallyGlob(Path base, boolean baseIsDir, int idx, GlobTaskContext context)
        throws IOException {
      if (idx == context.patternParts.length) { // Base case.
        maybeAddResult(context, base, baseIsDir);
        return;
      }

      // Do an early readdir() call here if the pattern contains a wildcard (* or ?). The reason is
      // that we'll do so later anyway and doing this early avoids an additional stat to determine
      // the existence of a build file as part of the shouldTraverseDirectory() call below (globs
      // will no recurse into sub-packages, i.e. directories that contain a build file). This
      // optimizes for the common case where there is no build file in the sub directory.
      String pattern = context.patternParts[idx];
      boolean patternContainsWildcard = pattern.contains("*") || pattern.contains("?");
      Collection<Dirent> dents = null;
      if (baseIsDir && patternContainsWildcard) {
        dents = context.syscalls.readdir(base);
      }

      if (baseIsDir && !context.pathDiscriminator.shouldTraverseDirectory(base)) {
        if (areAllRemainingPatternsDoubleStar(context, idx)) {
          // For SUBPACKAGES, we encounter this when all remaining patterns in the glob expression
          // are `**`s. In that case we should include the subpackage's PathFragment (relative to
          // the package fragment) in the matching results.
          maybeAddResult(context, base, baseIsDir);
        }
        return;
      }

      if (!baseIsDir) {
        // Nothing to find here.
        return;
      }

      // ** is special: it can match nothing at all.
      // For example, x/** matches x, **/y matches y, and x/**/y matches x/y.
      if (isRecursivePattern(pattern)) {
        context.queueGlob(base, baseIsDir, idx + 1);
      }

      if (!patternContainsWildcard) {
        // We do not need to do a readdir in this case, just a stat.
        Path child = base.getChild(pattern);
        FileStatus status = context.syscalls.statIfFound(child, Symlinks.FOLLOW);
        if (status == null || (!status.isDirectory() && !status.isFile())) {
          // The file is a dangling symlink, fifo, does not exist, etc.
          return;
        }

        context.queueGlob(child, status.isDirectory(), idx + 1);
        return;
      }

      for (Dirent dent : dents) {
        Dirent.Type childType = dent.getType();
        if (childType == Dirent.Type.UNKNOWN) {
          // The file is a special file (fifo, etc.). No need to even match against the pattern.
          continue;
        }
        if (matches(pattern, dent.getName(), cache)) {
          Path child = base.getChild(dent.getName());

          if (childType == Dirent.Type.SYMLINK) {
            processSymlink(child, idx, context);
          } else {
            processFileOrDirectory(child, childType == Dirent.Type.DIRECTORY, idx, context);
          }
        }
      }
    }

    private void maybeAddResult(GlobTaskContext context, Path base, boolean isDirectory) {
      if (context.pathDiscriminator.shouldIncludePathInResult(base, isDirectory)) {
        results.add(base);
      }
    }

    private static boolean areAllRemainingPatternsDoubleStar(
        GlobTaskContext context, int startIdx) {
      return Arrays.stream(context.patternParts, startIdx, context.patternParts.length)
          .allMatch("**"::equals);
    }

    /**
     * Process symlinks asynchronously. If we should used readdir(..., Symlinks.FOLLOW), that would
     * result in a sequential symlink resolution with many file system implementations. If the
     * underlying file system is networked and a single directory contains many symlinks, that can
     * lead to substantial slowness.
     */
    private void processSymlink(Path path, int idx, GlobTaskContext context) {
      context.queueTask(
          () -> {
            try {
              FileStatus status = context.syscalls.statIfFound(path, Symlinks.FOLLOW);
              if (status != null) {
                processFileOrDirectory(path, status.isDirectory(), idx, context);
              }
            } catch (IOException e) {
              ioException.compareAndSet(null, e);
            }
          });
    }

    private void processFileOrDirectory(
        Path path, boolean isDir, int idx, GlobTaskContext context) {
      boolean isRecursivePattern = isRecursivePattern(context.patternParts[idx]);
      if (isDir) {
        context.queueGlob(path, /* baseIsDir= */ true, idx + (isRecursivePattern ? 0 : 1));
      } else if (idx + 1 == context.patternParts.length) {
        maybeAddResult(context, path, /* isDirectory= */ false);
      }
    }
  }

  /**
   * Filters out exclude patterns from a Set of paths. Common cases such as wildcard-free patterns
   * or suffix patterns are special-cased to make this function efficient.
   */
  public static void removeExcludes(Set<String> paths, Collection<String> excludes)
      throws BadPattern {
    ArrayList<String> complexPatterns = new ArrayList<>(excludes.size());
    Map<String, List<String>> starstarSlashStarHeadTailPairs = new HashMap<>();
    for (String exclude : excludes) {
      if (isWildcardFree(exclude)) {
        paths.remove(exclude);
        continue;
      }
      int patternPos = exclude.indexOf("**/*");
      if (patternPos != -1) {
        String head = exclude.substring(0, patternPos);
        String tail = exclude.substring(patternPos + 4);
        if (isWildcardFree(head) && isWildcardFree(tail)) {
          starstarSlashStarHeadTailPairs.computeIfAbsent(head, h -> new ArrayList<>()).add(tail);
          continue;
        }
      }
      complexPatterns.add(exclude);
    }
    for (Map.Entry<String, List<String>> headTailPair : starstarSlashStarHeadTailPairs.entrySet()) {
      paths.removeIf(
          path -> {
            if (path.startsWith(headTailPair.getKey())) {
              for (String tail : headTailPair.getValue()) {
                if (path.endsWith(tail)) {
                  return true;
                }
              }
            }
            return false;
          });
    }
    if (complexPatterns.isEmpty()) {
      return;
    }
    // TODO(adonovan): validate pattern unconditionally (potentially breaking change).
    List<String[]> splitPatterns = checkAndSplitPatterns(complexPatterns);
    HashMap<String, Pattern> patternCache = new HashMap<>();
    paths.removeIf(
        path -> {
          String[] segments = Iterables.toArray(Splitter.on('/').split(path), String.class);
          for (String[] splitPattern : splitPatterns) {
            if (matchesPattern(splitPattern, segments, 0, 0, patternCache)) {
              return true;
            }
          }
          return false;
        });
  }

  /** Returns true if {@code pattern} matches {@code path} starting from the given segments. */
  private static boolean matchesPattern(
      String[] pattern, String[] path, int i, int j, Map<String, Pattern> patternCache) {
    if (i == pattern.length) {
      return j == path.length;
    }
    if (pattern[i].equals("**")) {
      return matchesPattern(pattern, path, i + 1, j, patternCache)
          || (j < path.length && matchesPattern(pattern, path, i, j + 1, patternCache));
    }
    if (j == path.length) {
      return false;
    }
    if (matches(pattern[i], path[j], patternCache)) {
      return matchesPattern(pattern, path, i + 1, j + 1, patternCache);
    }
    return false;
  }

  private static boolean isWildcardFree(String pattern) {
    return !pattern.contains("*") && !pattern.contains("?");
  }
}
