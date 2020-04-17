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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Splitter;
import com.google.common.base.Throwables;
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
import java.io.IOException;
import java.util.ArrayList;
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
      boolean excludeDirectories,
      Predicate<Path> dirPred,
      FilesystemCalls syscalls,
      Executor executor)
      throws IOException, InterruptedException, BadPattern {
    GlobVisitor visitor = new GlobVisitor(executor);
    return visitor.glob(base, patterns, excludeDirectories, dirPred, syscalls);
  }

  private static List<Path> globInternalUninterruptible(
      Path base,
      Collection<String> patterns,
      boolean excludeDirectories,
      Predicate<Path> dirPred,
      FilesystemCalls syscalls,
      Executor executor)
      throws IOException, BadPattern {
    GlobVisitor visitor = new GlobVisitor(executor);
    return visitor.globUninterruptible(base, patterns, excludeDirectories, dirPred, syscalls);
  }

  private static long globInternalAndReturnNumGlobTasksForTesting(
      Path base,
      Collection<String> patterns,
      boolean excludeDirectories,
      Predicate<Path> dirPred,
      FilesystemCalls syscalls,
      Executor executor)
      throws IOException, InterruptedException, BadPattern {
    GlobVisitor visitor = new GlobVisitor(executor);
    visitor.glob(base, patterns, excludeDirectories, dirPred, syscalls);
    return visitor.getNumGlobTasksForTesting();
  }

  private static Future<List<Path>> globAsyncInternal(
      Path base,
      Collection<String> patterns,
      boolean excludeDirectories,
      Predicate<Path> dirPred,
      FilesystemCalls syscalls,
      Executor executor)
      throws BadPattern {
    Preconditions.checkNotNull(executor, "%s %s", base, patterns);
    return new GlobVisitor(executor)
        .globAsync(base, patterns, excludeDirectories, dirPred, syscalls);
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
      switch(c) {
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
        case '(':
        case ')':
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
        default:
          regexp.append(c);
          break;
      }
    }
    return Pattern.compile(regexp.toString());
  }

  /**
   * Filesystem calls required for glob().
   */
  public interface FilesystemCalls {
    /** Get directory entries and their types. Does not follow symlinks. */
    Collection<Dirent> readdir(Path path) throws IOException;

    /** Return the stat() for the given path, or null. */
    FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException;

    /**
     * Return the type of a specific file. This may be answered using stat() or readdir(). Returns
     * null if the path does not exist.
     */
    Dirent.Type getType(Path path, Symlinks symlinks) throws IOException;
  }

  public static final FilesystemCalls DEFAULT_SYSCALLS =
      new FilesystemCalls() {
        @Override
        public Collection<Dirent> readdir(Path path) throws IOException {
          return path.readdir(Symlinks.NOFOLLOW);
        }

        @Override
        public FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException {
          return path.statIfFound(symlinks);
        }

        @Override
        public Dirent.Type getType(Path path, Symlinks symlinks) throws IOException {
          return statusToDirentType(statIfFound(path, symlinks));
        }
      };

  public static Dirent.Type statusToDirentType(FileStatus status) {
    if (status == null) {
      return null;
    } else if (status.isFile()) {
      return Dirent.Type.FILE;
    } else if (status.isDirectory()) {
      return Dirent.Type.DIRECTORY;
    } else if (status.isSymbolicLink()) {
      return Dirent.Type.SYMLINK;
    }
    return Dirent.Type.UNKNOWN;
  }

  public static final AtomicReference<FilesystemCalls> DEFAULT_SYSCALLS_REF =
      new AtomicReference<>(DEFAULT_SYSCALLS);

  public static Builder forPath(Path path) {
    return new Builder(path);
  }

  /**
   * Builder class for UnixGlob.
   *
   *
   */
  public static class Builder {
    private Path base;
    private List<String> patterns;
    private boolean excludeDirectories;
    private Predicate<Path> pathFilter;
    private Executor executor;
    private AtomicReference<? extends FilesystemCalls> syscalls =
        new AtomicReference<>(DEFAULT_SYSCALLS);

    /**
     * Creates a glob builder with the given base path.
     */
    public Builder(Path base) {
      this.base = base;
      this.patterns = Lists.newArrayList();
      this.excludeDirectories = false;
      this.pathFilter = Predicates.alwaysTrue();
    }

    /**
     * Adds a pattern to include to the glob builder.
     *
     * <p>For a description of the syntax of the patterns, see {@link UnixGlob}.
     */
    public Builder addPattern(String pattern) {
      this.patterns.add(pattern);
      return this;
    }

    /**
     * Adds a pattern to include to the glob builder.
     *
     * <p>For a description of the syntax of the patterns, see {@link UnixGlob}.
     */
    public Builder addPatterns(String... patterns) {
      Collections.addAll(this.patterns, patterns);
      return this;
    }

    /**
     * Adds a pattern to include to the glob builder.
     *
     * <p>For a description of the syntax of the patterns, see {@link UnixGlob}.
     */
    public Builder addPatterns(Collection<String> patterns) {
      this.patterns.addAll(patterns);
      return this;
    }

    /**
     * Sets the FilesystemCalls interface to use on this glob().
     */
    public Builder setFilesystemCalls(AtomicReference<? extends FilesystemCalls> syscalls) {
      this.syscalls = (syscalls == null)
          ? new AtomicReference<FilesystemCalls>(DEFAULT_SYSCALLS)
          : syscalls;
      return this;
    }

    /**
     * If set to true, directories are not returned in the glob result.
     */
    public Builder setExcludeDirectories(boolean excludeDirectories) {
      this.excludeDirectories = excludeDirectories;
      return this;
    }

    /**
     * Sets the executor to use for parallel glob evaluation. If unset, evaluation is done
     * in-thread.
     */
    public Builder setExecutor(Executor pool) {
      this.executor = pool;
      return this;
    }


    /**
     * If set, the given predicate is called for every directory
     * encountered. If it returns false, the corresponding item is not
     * returned in the output and directories are not traversed either.
     */
    public Builder setDirectoryFilter(Predicate<Path> pathFilter) {
      this.pathFilter = pathFilter;
      return this;
    }

    /** Executes the glob. */
    public List<Path> glob() throws IOException, BadPattern {
      return globInternalUninterruptible(
          base, patterns, excludeDirectories, pathFilter, syscalls.get(), executor);
    }

    /**
     * Executes the glob and returns the result.
     *
     * @throws InterruptedException if the thread is interrupted.
     */
    public List<Path> globInterruptible() throws IOException, InterruptedException, BadPattern {
      return globInternal(base, patterns, excludeDirectories, pathFilter, syscalls.get(), executor);
    }

    @VisibleForTesting
    public long globInterruptibleAndReturnNumGlobTasksForTesting()
        throws IOException, InterruptedException, BadPattern {
      return globInternalAndReturnNumGlobTasksForTesting(
          base, patterns, excludeDirectories, pathFilter, syscalls.get(), executor);
    }

    /**
     * Executes the glob asynchronously. {@link #setExecutor} must have been called already with a
     * non-null argument.
     */
    public Future<List<Path>> globAsync() throws BadPattern {
      return globAsyncInternal(
          base, patterns, excludeDirectories, pathFilter, syscalls.get(), executor);
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
     * relative to {@code base}. Directories are traversed if and only if they match {@code
     * dirPred}. The predicate is also called for the root of the traversal. The order of the
     * returned list is unspecified.
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
        boolean excludeDirectories,
        Predicate<Path> dirPred,
        FilesystemCalls syscalls)
        throws IOException, InterruptedException, BadPattern {
      try {
        return globAsync(base, patterns, excludeDirectories, dirPred, syscalls).get();
      } catch (ExecutionException e) {
        Throwable cause = e.getCause();
        Throwables.propagateIfPossible(cause, IOException.class);
        throw new RuntimeException(e);
      }
    }

    List<Path> globUninterruptible(
        Path base,
        Collection<String> patterns,
        boolean excludeDirectories,
        Predicate<Path> dirPred,
        FilesystemCalls syscalls)
        throws IOException, BadPattern {
      try {
        return Uninterruptibles.getUninterruptibly(
            globAsync(base, patterns, excludeDirectories, dirPred, syscalls));
      } catch (ExecutionException e) {
        Throwable cause = e.getCause();
        Throwables.propagateIfPossible(cause, IOException.class);
        Throwables.propagateIfPossible(cause, BadPattern.class);
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
        boolean excludeDirectories,
        Predicate<Path> dirPred,
        FilesystemCalls syscalls)
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
          GlobTaskContext context = numRecursivePatterns > 1
              ? new RecursiveGlobTaskContext(splitPattern, excludeDirectories, dirPred, syscalls)
              : new GlobTaskContext(splitPattern, excludeDirectories, dirPred, syscalls);
          context.queueGlob(base, baseStat.isDirectory(), 0);
        }
      } finally {
        decrementAndCheckDone();
      }

      return result;
    }

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

    /** Should only be called by link {@GlobTaskContext}. */
    private void queueGlob(final Path base, final boolean baseIsDir, final int idx,
        final GlobTaskContext context) {
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
                  "%s glob(include=[%s], exclude_directories=%s)",
                  base.getPathString(),
                  "\"" + Joiner.on("\", \"").join(context.patternParts) + "\"",
                  context.excludeDirectories);
            }
          });
    }

    /** Should only be called by link {@GlobTaskContext}. */
    private void queueTask(Runnable runnable) {
      enqueue(runnable);
    }

    protected void enqueue(final Runnable r) {
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

    protected void cancel() {
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
      private final boolean excludeDirectories;
      private final Predicate<Path> dirPred;
      private final FilesystemCalls syscalls;

      GlobTaskContext(
          String[] patternParts,
          boolean excludeDirectories,
          Predicate<Path> dirPred,
          FilesystemCalls syscalls) {
        this.patternParts = patternParts;
        this.excludeDirectories = excludeDirectories;
        this.dirPred = dirPred;
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
          boolean excludeDirectories,
          Predicate<Path> dirPred,
          FilesystemCalls syscalls) {
        super(patternParts, excludeDirectories, dirPred, syscalls);
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
      if (baseIsDir && !context.dirPred.apply(base)) {
        return;
      }

      if (idx == context.patternParts.length) { // Base case.
        if (!(context.excludeDirectories && baseIsDir)) {
          results.add(base);
        }

        return;
      }

      if (!baseIsDir) {
        // Nothing to find here.
        return;
      }

      String pattern = context.patternParts[idx];

      // ** is special: it can match nothing at all.
      // For example, x/** matches x, **/y matches y, and x/**/y matches x/y.
      if (isRecursivePattern(pattern)) {
        context.queueGlob(base, baseIsDir, idx + 1);
      }

      if (!pattern.contains("*") && !pattern.contains("?")) {
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

      Collection<Dirent> dents = context.syscalls.readdir(base);
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
              // Intentionally empty. Just ignore symlinks that cannot be stat'ed to leave
              // historical behavior of readdir(..., Symlinks.FOLLOW).
            }
          });
    }

    private void processFileOrDirectory(
        Path path, boolean isDir, int idx, GlobTaskContext context) {
      boolean isRecursivePattern = isRecursivePattern(context.patternParts[idx]);
      if (isDir) {
        context.queueGlob(path, /* baseIsDir= */ true, idx + (isRecursivePattern ? 0 : 1));
      } else if (idx + 1 == context.patternParts.length) {
        results.add(path);
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
