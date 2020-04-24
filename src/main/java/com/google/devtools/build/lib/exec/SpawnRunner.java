// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FutureSpawn;
import com.google.devtools.build.lib.actions.LostInputsExecException;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.time.Duration;
import java.util.SortedMap;
import javax.annotation.Nullable;

/**
 * A runner for spawns. Implementations can execute spawns on the local machine as a subprocess with
 * or without sandboxing, on a remote machine, or only consult a remote cache.
 *
 * <h2>Environment Variables</h2>
 *
 * <ul>
 *   <li>Implementations MUST set the specified environment variables.
 *   <li>Implementations MAY add TMPDIR as an additional env variable, if it is not set already.
 *   <li>If an implementation sets TMPDIR, it MUST be set to an absolute path.
 *   <li>Implementations MUST NOT add any other environment variables.
 * </ul>
 *
 * <h2>Command line</h2>
 *
 * <ul>
 *   <li>Implementations MUST use the specified command line unmodified by default.
 *   <li>Implementations MAY modify the specified command line if explicitly requested by the user.
 * </ul>
 *
 * <h2>Process</h2>
 *
 * <ul>
 *   <li>Implementations MUST be thread-safe.
 *   <li>Implementations MUST ensure that all child processes (including transitive) exit in all
 *       cases, including successful completion, interruption, and timeout
 *   <li>Implementations MUST return the exit code as observed from the subprocess if the subprocess
 *       exits naturally; they MUST not throw an exception for non-zero exit codes
 *   <li>Implementations MUST be interruptible; they MUST throw {@link InterruptedException} from
 *       {@link #exec} when interrupted
 *   <li>Implementations MUST apply the specified timeout to the execution of the subprocess
 *       <ul>
 *         <li>If no timeout is specified, the implementation MAY apply an implementation-specific
 *             timeout
 *         <li>If the specified timeout is larger than an implementation-dependent maximum, then the
 *             implementation MUST throw {@link IllegalArgumentException}; it MUST not silently
 *             change the timeout to a smaller value
 *         <li>If the timeout is exceeded, the implementation MUST throw TimeoutException, with the
 *             timeout that was applied to the subprocess (TODO)
 *       </ul>
 * </ul>
 *
 * <h2>Optimistic Concurrency</h2>
 *
 * Bazel may choose to execute a spawn using multiple {@link SpawnRunner} implementations
 * simultaneously in order to minimize total latency. This is especially useful for builds with few
 * actions where remotely executing the actions incurs high round trip times.
 *
 * <ul>
 *   <li>All implementations MUST call {@link SpawnExecutionContext#lockOutputFiles} before writing
 *       to any of the output files, but may write to stdout and stderr without calling it. Instead,
 *       all callers must provide temporary locations for stdout & stderr if they ever call multiple
 *       {@link SpawnRunner} implementations concurrently. Spawn runners that use the local machine
 *       MUST either call it before starting the subprocess, or ensure that subprocesses write to
 *       temporary locations (for example by running in a mount namespace) and then copy or move the
 *       outputs into place.
 *   <li>Implementations SHOULD delay calling {@link SpawnExecutionContext#lockOutputFiles} until
 *       just before writing.
 * </ul>
 */
public interface SpawnRunner {
  /**
   * Used to report progress on the current spawn. This is mainly used to report the current state
   * of the subprocess to the user, but may also be used to trigger parallel execution. For example,
   * a dynamic scheduler may use the signal that there was a cache miss to start parallel execution
   * of the same Spawn - also see the {@link SpawnRunner} documentation section on "optimistic
   * concurrency".
   *
   * <p>{@link SpawnRunner} implementations should post a progress status before any potentially
   * long-running operation.
   */
  enum ProgressStatus {
    /** Spawn is waiting for local or remote resources to become available. */
    SCHEDULING,

    /** The {@link SpawnRunner} is looking for a cache hit. */
    CHECKING_CACHE,

    /**
     * Resources are acquired, and there was probably no cache hit. This MUST be posted before
     * attempting to execute the subprocess.
     *
     * <p>Caching {@link SpawnRunner} implementations should only post this after a failed cache
     * lookup, but may post this if cache lookup and execution happen within the same step, e.g. as
     * part of a single RPC call with no mechanism to report cache misses.
     */
    EXECUTING,

    /** Downloading outputs from a remote machine. */
    DOWNLOADING
  }

  /**
   * A context that binds a {@link Spawn} to a {@link SpawnRunner}.
   *
   * <p>This interface may change without notice.
   *
   * <p>Implementations must be at least thread-compatible, i.e., they must be safe as long as each
   * instance is only used within a single thread. Different instances of the same class may be used
   * by different threads, so they MUST not call any shared non-thread-safe objects.
   */
  interface SpawnExecutionContext {
    /**
     * Returns a unique id for this spawn, to be used for logging. Note that a single spawn may be
     * passed to multiple {@link SpawnRunner} implementations, so any log entries should also
     * contain the identity of the spawn runner implementation.
     */
    int getId();

    /**
     * Prefetches the Spawns input files to the local machine. There are cases where Bazel runs on a
     * network file system, and prefetching the files in parallel is a significant performance win.
     * This should only be called by local strategies when local execution is imminent.
     *
     * <p>Should be called with the equivalent of: <code>
     * policy.prefetchInputs(
     *      Iterables.filter(policy.getInputMapping().values(), Predicates.notNull()));
     * </code>
     *
     * <p>Note in particular that {@link #getInputMapping} may return {@code null} values, but this
     * method does not accept {@code null} values.
     *
     * <p>The reason why this method requires passing in the inputs is that getInputMapping may be
     * slow to compute, so if the implementation already called it, we don't want to compute it
     * again. I suppose we could require implementations to memoize getInputMapping (but not compute
     * it eagerly), and that may change in the future.
     */
    void prefetchInputs() throws IOException, InterruptedException;

    /**
     * The input file metadata cache for this specific spawn, which can be used to efficiently
     * obtain file digests and sizes.
     */
    MetadataProvider getMetadataProvider();

    /** An artifact expander. */
    // TODO(ulfjack): This is only used for the sandbox runners to compute a set of empty
    // directories. We shouldn't have this and the getInputMapping method; maybe there's a way to
    // unify the two? Alternatively, maybe the input mapping should (optionally?) contain
    // directories? Or maybe we need a separate method to return the set of directories?
    ArtifactExpander getArtifactExpander();

    /** The {@link ArtifactPathResolver} to use when directly writing output files. */
    default ArtifactPathResolver getPathResolver() {
      return ArtifactPathResolver.IDENTITY;
    }

    /**
     * All implementations must call this method before writing to the provided stdout / stderr or
     * to any of the output file locations. This method is used to coordinate - implementations must
     * throw an {@link InterruptedException} for all but one caller.
     */
    void lockOutputFiles() throws InterruptedException, IOException;

    /**
     * Returns whether this spawn may be executing concurrently under multiple spawn runners. If so,
     * {@link #lockOutputFiles} may raise {@link InterruptedException}.
     */
    boolean speculating();

    /** Returns the timeout that should be applied for the given {@link Spawn} instance. */
    Duration getTimeout();

    /** The files to which to write stdout and stderr. */
    FileOutErr getFileOutErr();

    SortedMap<PathFragment, ActionInput> getInputMapping(boolean expandTreeArtifactsInRunfiles)
        throws IOException;

    /** Reports a progress update to the Spawn strategy. */
    void report(ProgressStatus state, String name);

    /**
     * Returns a {@link MetadataInjector} that allows a caller to inject metadata about spawn
     * outputs that are stored remotely.
     */
    MetadataInjector getMetadataInjector();

    /**
     * Returns the context registered for the given identifying type or {@code null} if none was
     * registered.
     */
    @Nullable
    <T extends ActionContext> T getContext(Class<T> identifyingType);

    /** Returns whether rewinding is enabled. */
    boolean isRewindingEnabled();

    /** Throws if rewinding is enabled and lost inputs have been detected. */
    void checkForLostInputs() throws LostInputsExecException;
  }

  /**
   * Run the given spawn asynchronously. The default implementation is synchronous for migration.
   *
   * @param spawn the spawn to run
   * @param context the spawn execution context
   * @return the result from running the spawn
   * @throws InterruptedException if the calling thread was interrupted, or if the runner could not
   *     lock the output files (see {@link SpawnExecutionContext#lockOutputFiles()})
   * @throws IOException if something went wrong reading or writing to the local file system
   * @throws ExecException if the request is malformed
   */
  default FutureSpawn execAsync(Spawn spawn, SpawnExecutionContext context)
      throws InterruptedException, IOException, ExecException {
    // TODO(ulfjack): Remove this default implementation. [exec-async]
    return FutureSpawn.immediate(exec(spawn, context));
  }

  /**
   * Run the given spawn.
   *
   * @param spawn the spawn to run
   * @param context the spawn execution context
   * @return the result from running the spawn
   * @throws InterruptedException if the calling thread was interrupted, or if the runner could not
   *     lock the output files (see {@link SpawnExecutionContext#lockOutputFiles()})
   * @throws IOException if something went wrong reading or writing to the local file system
   * @throws ExecException if the request is malformed
   */
  SpawnResult exec(Spawn spawn, SpawnExecutionContext context)
      throws InterruptedException, IOException, ExecException;

  /** Returns whether this SpawnRunner supports executing the given Spawn. */
  boolean canExec(Spawn spawn);

  /** Returns the name of the SpawnRunner. */
  String getName();

  /**
   * Removes any files or directories that this spawn runner may have put in the sandbox base.
   *
   * <p>It is important that this function only removes entries that may have been generated by this
   * build, not any possible entries that a future build may generate.
   *
   * @param sandboxBase path to the base of the sandbox tree where the spawn runner may have created
   *     entries
   * @param treeDeleter scheduler for tree deletions
   * @throws IOException if there are problems deleting the entries
   */
  default void cleanupSandboxBase(Path sandboxBase, TreeDeleter treeDeleter) throws IOException {}
}
