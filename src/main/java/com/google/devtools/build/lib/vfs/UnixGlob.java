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

import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Splitter;
import com.google.common.base.Throwables;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ForwardingListenableFuture;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.Preconditions;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;

/**
 * Implementation of a subset of UNIX-style file globbing, expanding "*" and "?" as wildcards, but
 * not [a-z] ranges.
 *
 * <p><code>**</code> gets special treatment in include patterns. If it is used as a complete path
 * segment it matches the filenames in subdirectories recursively.
 */
public final class UnixGlob {
  private UnixGlob() {}

  private static List<Path> globInternal(Path base, Collection<String> patterns,
                                         boolean excludeDirectories,
                                         Predicate<Path> dirPred,
                                         boolean checkForInterruption,
                                         FilesystemCalls syscalls,
                                         ThreadPoolExecutor threadPool)
      throws IOException, InterruptedException {
    GlobVisitor visitor = (threadPool == null)
        ? new GlobVisitor(checkForInterruption)
        : new GlobVisitor(threadPool, checkForInterruption);
    return visitor.glob(base, patterns, excludeDirectories, dirPred, syscalls);
  }

  private static Future<List<Path>> globAsyncInternal(Path base, Collection<String> patterns,
                                                      boolean excludeDirectories,
                                                      Predicate<Path> dirPred,
                                                      FilesystemCalls syscalls,
                                                      boolean checkForInterruption,
                                                      ThreadPoolExecutor threadPool) {
    Preconditions.checkNotNull(threadPool, "%s %s", base, patterns);
    try {
      return new GlobVisitor(threadPool, checkForInterruption)
          .globAsync(base, patterns, excludeDirectories, dirPred, syscalls);
    } catch (IOException e) {
      // We are evaluating asynchronously, so no exceptions should be thrown until the future is
      // retrieved.
      throw new IllegalStateException(e);
    }
  }

  /**
   * Checks that each pattern is valid, splits it into segments and checks
   * that each segment contains only valid wildcards.
   *
   * @return list of segment arrays
   */
  private static List<String[]> checkAndSplitPatterns(Collection<String> patterns) {
    List<String[]> list = Lists.newArrayListWithCapacity(patterns.size());
    for (String pattern : patterns) {
      String error = checkPatternForError(pattern);
      if (error != null) {
        throw new IllegalArgumentException(error + " (in glob pattern '" + pattern + "')");
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
    for (int i = 0; i < pattern.length(); i++) {
      char c = pattern.charAt(i);
      switch (c) {
        case '(': case ')':
        case '{': case '}':
        case '[': case ']':
        return "illegal character '" + c + "'";
      }
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

  /**
   * Calls {@link #matches(String, String, Cache) matches(pattern, str, null)}
   */
  public static boolean matches(String pattern, String str) {
    return matches(pattern, str, null);
  }

  /**
   * Returns whether {@code str} matches the glob pattern {@code pattern}. This
   * method may use the {@code patternCache} to speed up the matching process.
   *
   * @param pattern a glob pattern
   * @param str the string to match
   * @param patternCache a cache from patterns to compiled Pattern objects, or
   *        {@code null} to skip caching
   */
  public static boolean matches(String pattern, String str,
      Cache<String, Pattern> patternCache) {
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

    Pattern regex = patternCache == null ? null : patternCache.getIfPresent(pattern);
    if (regex == null) {
      regex = makePatternFromWildcard(pattern);
      if (patternCache != null) {
        patternCache.put(pattern, regex);
      }
    }
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
    for(int i = 0, len = pattern.length(); i < len; i++) {
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
        //escape the regexp special characters that are allowed in wildcards
        case '^': case '$': case '|': case '+':
        case '{': case '}': case '[': case ']':
        case '\\': case '.':
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
    /**
     * Get directory entries and their types.
     */
    Collection<Dirent> readdir(Path path, Symlinks symlinks) throws IOException;

    /**
     * Return the stat() for the given path, or null.
     */
    FileStatus statNullable(Path path, Symlinks symlinks);
  }

  public static FilesystemCalls DEFAULT_SYSCALLS = new FilesystemCalls() {
    @Override
    public Collection<Dirent> readdir(Path path, Symlinks symlinks) throws IOException {
      return path.readdir(symlinks);
    }

    @Override
    public FileStatus statNullable(Path path, Symlinks symlinks) {
      return path.statNullable(symlinks);
    }
  };

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
    private ThreadPoolExecutor threadPool;
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
     * Sets the threadpool to use for parallel glob evaluation.
     * If unset, evaluation is done in-thread.
     */
    public Builder setThreadPool(ThreadPoolExecutor pool) {
      this.threadPool = pool;
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

    /**
     * Executes the glob.
     */
    public List<Path> glob() throws IOException {
      try {
        return globInternal(base, patterns, excludeDirectories, pathFilter, false,
            syscalls.get(), threadPool);
      } catch (InterruptedException e) {
        // cannot happen, since we told globInternal not to throw
        throw new IllegalStateException(e);
      }
    }

    /**
     * Executes the glob.
     *
     * @throws InterruptedException if the thread is interrupted.
     */
    public List<Path> globInterruptible() throws IOException, InterruptedException {
      return globInternal(base, patterns, excludeDirectories, pathFilter, true,
          syscalls.get(), threadPool);
    }

    /**
     * Executes the glob asynchronously. {@link #setThreadPool} must have been called already with a
     * non-null argument.
     *
     * @param checkForInterrupt if the returned future may throw InterruptedException.
     */
    public Future<List<Path>> globAsync(boolean checkForInterrupt) {
      return globAsyncInternal(base, patterns, excludeDirectories, pathFilter,
          syscalls.get(), checkForInterrupt, threadPool);
    }
  }

  /**
   * Adapts the result of the glob visitation as a Future.
   */
  private static class GlobFuture extends ForwardingListenableFuture<List<Path>> {
    private final GlobVisitor visitor;
    private final boolean checkForInterrupt;
    private final SettableFuture<List<Path>> delegate = SettableFuture.create();

    public GlobFuture(GlobVisitor visitor, boolean interruptible) {
      this.visitor = visitor;
      this.checkForInterrupt = interruptible;
    }

    @Override
    public List<Path> get() throws InterruptedException, ExecutionException {
      return checkForInterrupt ? super.get() : Uninterruptibles.getUninterruptibly(delegate());
    }

    @Override
    protected ListenableFuture<List<Path>> delegate() {
      return delegate;
    }

    public void setException(IOException exception) {
      delegate.setException(exception);
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
    private final Cache<String, Pattern> cache = CacheBuilder.newBuilder().build(
        new CacheLoader<String, Pattern>() {
            @Override
            public Pattern load(String wildcard) {
              return makePatternFromWildcard(wildcard);
            }
          });

    private final GlobFuture result;
    private final ThreadPoolExecutor executor;
    private final AtomicLong pendingOps = new AtomicLong(0);
    private final AtomicReference<IOException> failure = new AtomicReference<>();
    private volatile boolean canceled = false;

    public GlobVisitor(ThreadPoolExecutor executor, boolean failFastOnInterrupt) {
      this.executor = executor;
      this.result = new GlobFuture(this, failFastOnInterrupt);
    }

    public GlobVisitor(boolean failFastOnInterrupt) {
      this(null, failFastOnInterrupt);
    }

    /**
     * Performs wildcard globbing: returns the sorted list of filenames that match any of
     * {@code patterns} relative to {@code base}. Directories are traversed if and only if they
     * match {@code dirPred}. The predicate is also called for the root of the traversal.
     *
     * <p>Patterns may include "*" and "?", but not "[a-z]".
     *
     * <p><code>**</code> gets special treatment in include patterns. If it is
     * used as a complete path segment it matches the filenames in
     * subdirectories recursively.
     *
     * @throws IllegalArgumentException if any glob pattern
     *         {@linkplain #checkPatternForError(String) contains errors} or if any include pattern
     *         segment contains <code>**</code> but not equal to it.
     */
    public List<Path> glob(Path base, Collection<String> patterns,
                           boolean excludeDirectories, Predicate<Path> dirPred,
                           FilesystemCalls syscalls)
        throws IOException, InterruptedException {
      try {
        return globAsync(base, patterns, excludeDirectories, dirPred, syscalls).get();
      } catch (ExecutionException e) {
        Throwable cause = e.getCause();
        Throwables.propagateIfPossible(cause, IOException.class);
        throw new RuntimeException(e);
      }
    }

    public Future<List<Path>> globAsync(Path base, Collection<String> patterns,
        boolean excludeDirectories, Predicate<Path> dirPred, FilesystemCalls syscalls)
            throws IOException {

      FileStatus baseStat = syscalls.statNullable(base, Symlinks.FOLLOW);
      if (baseStat == null || patterns.isEmpty()) {
        return Futures.immediateFuture(Collections.<Path>emptyList());
      }

      List<String[]> splitPatterns = checkAndSplitPatterns(patterns);

      // We do a dumb loop, even though it will likely duplicate work
      // (e.g., readdir calls). In order to optimize, we would need
      // to keep track of which patterns shared sub-patterns and which did not
      // (for example consider the glob [*/*.java, sub/*.java, */*.txt]).
      pendingOps.incrementAndGet();
      try {
        for (String[] splitPattern : splitPatterns) {
          queueGlob(base, baseStat.isDirectory(), splitPattern, 0, excludeDirectories, results,
              cache, dirPred, syscalls);
        }
      } finally {
        decrementAndCheckDone();
      }

      return result;
    }

    private void queueGlob(final Path base, final boolean baseIsDir,
        final String[] patternParts, final int idx,
        final boolean excludeDirectories,
        final Collection<Path> results, final Cache<String, Pattern> cache,
        final Predicate<Path> dirPred, final FilesystemCalls syscalls) throws IOException {
      enqueue(new Runnable() {
        @Override
        public void run() {
          Profiler.instance().startTask(ProfilerTask.VFS_GLOB, this);
          try {
            reallyGlob(base, baseIsDir, patternParts, idx, excludeDirectories, results, cache,
                dirPred, syscalls);
          } catch (IOException e) {
            failure.set(e);
          } finally {
            Profiler.instance().completeTask(ProfilerTask.VFS_GLOB);
          }
        }

        @Override
        public String toString() {
          return String.format(
                  "%s glob(include=[%s], exclude_directories=%s)",
                  base.getPathString(),
                  "\"" + Joiner.on("\", \"").join(patternParts) + "\"",
                  excludeDirectories);
        }
      });
    }

    protected void enqueue(final Runnable r) {
      pendingOps.incrementAndGet();

      Runnable wrapped = new Runnable() {
        @Override
        public void run() {
          try {
            if (!canceled && failure.get() == null) {
              r.run();
            }
          } finally {
            decrementAndCheckDone();
          }
        }
      };

      if (executor == null) {
        wrapped.run();
      } else {
        executor.execute(wrapped);
      }
    }

    protected void cancel() {
      this.canceled = true;
    }

    private void decrementAndCheckDone() {
      if (pendingOps.decrementAndGet() == 0) {
        // We get to 0 iff we are done all the relevant work. This is because we always increment
        // the pending ops count as we're enqueuing, and don't decrement until the task is complete
        // (which includes accounting for any additional tasks that one enqueues).
        if (canceled) {
          result.markCanceled();
        } else if (failure.get() != null) {
          result.setException(failure.get());
        } else {
          result.set(Ordering.<Path>natural().immutableSortedCopy(results));
        }
      }
    }

    /**
     * Expressed in Haskell:
     * <pre>
     *  reallyGlob base []     = { base }
     *  reallyGlob base [x:xs] = union { reallyGlob(f, xs) | f results "base/x" }
     * </pre>
     */
    private void reallyGlob(Path base, boolean baseIsDir, String[] patternParts, int idx,
        boolean excludeDirectories,
        Collection<Path> results, Cache<String, Pattern> cache,
        Predicate<Path> dirPred,
        FilesystemCalls syscalls) throws IOException {
      if (baseIsDir && !dirPred.apply(base)) {
        return;
      }

      if (idx == patternParts.length) { // Base case.
        if (!(excludeDirectories && baseIsDir)) {
          results.add(base);
        }

        return;
      }

      if (!baseIsDir) {
        // Nothing to find here.
        return;
      }

      final String pattern = patternParts[idx];

      // ** is special: it can match nothing at all.
      // For example, x/** matches x, **/y matches y, and x/**/y matches x/y.
      if ("**".equals(pattern)) {
        queueGlob(base, baseIsDir, patternParts, idx + 1, excludeDirectories, results, cache,
            dirPred, syscalls);
      }

      if (!pattern.contains("*") && !pattern.contains("?")) {
        // We do not need to do a readdir in this case, just a stat.
        Path child = base.getChild(pattern);
        FileStatus status = syscalls.statNullable(child, Symlinks.FOLLOW);
        if (status == null || (!status.isDirectory() && !status.isFile())) {
          // The file is a dangling symlink, fifo, does not exist, etc.
          return;
        }

        boolean childIsDir = status.isDirectory();

        queueGlob(child, childIsDir, patternParts, idx + 1, excludeDirectories, results, cache,
            dirPred, syscalls);
        return;
      }

      Collection<Dirent> dents = syscalls.readdir(base, Symlinks.FOLLOW);

      for (Dirent dent : dents) {
        Dirent.Type type = dent.getType();
        if (type == Dirent.Type.UNKNOWN) {
          // The file is a dangling symlink, fifo, etc.
          continue;
        }
        boolean childIsDir = (type == Dirent.Type.DIRECTORY);
        String text = dent.getName();
        Path child = base.getChild(text);

        if ("**".equals(pattern)) {
          // Recurse without shifting the pattern.
          if (childIsDir) {
            queueGlob(child, childIsDir, patternParts, idx, excludeDirectories, results, cache,
                dirPred, syscalls);
          }
        }
        if (matches(pattern, text, cache)) {
          // Recurse and consume one segment of the pattern.
          if (childIsDir) {
            queueGlob(child, childIsDir, patternParts, idx + 1, excludeDirectories, results, cache,
                dirPred, syscalls);
          } else {
            // Instead of using an async call, just repeat the base case above.
            if (idx + 1 == patternParts.length) {
              results.add(child);
            }
          }
        }
      }
    }
  }
}
