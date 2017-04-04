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

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.SortedMap;

/**
 * A runner for spawns. Implementations can execute spawns on the local machine as a subprocess with
 * or without sandboxing, on a remote machine, or only consult a remote cache.
 *
 * <h2>Environment Variables</h2>
 * <ul>
 *   <li>Implementations MUST set the specified environment variables.
 *   <li>Implementations MAY add TMPDIR as an additional env variable, if it is not set already.
 *   <li>If an implementation sets TMPDIR, it MUST be set to an absolute path.
 *   <li>Implementations MUST NOT add any other environment variables.
 * </ul>
 *
 * <h2>Command line</h2>
 * <ul>
 *   <li>Implementations MUST use the specified command line unmodified by default.
 *   <li>Implementations MAY modify the specified command line if explicitly requested by the user.
 * </ul>
 *
 * <h2>Process</h2>
 * <ul>
 *   <li>Implementations MUST ensure that all child processes (including transitive) exit in all
 *       cases, including successful completion, interruption, and timeout
 *   <li>Implementations MUST return the exit code as observed from the subprocess if the subprocess
 *       exits naturally; they MUST not throw an exception for non-zero exit codes
 *   <li>Implementations MUST be interruptible; they MUST throw {@link InterruptedException} from
 *       {@link #exec} when interrupted
 *   <li>Implementations MUST apply the specified timeout to the execution of the subprocess
 *     <ul>
 *       <li>If no timeout is specified, the implementation MAY apply an implementation-specific
 *           timeout
 *       <li>If the specified timeout is larger than an implementation-dependent maximum, then the
 *           implementation MUST throw {@link IllegalArgumentException}; it MUST not silently change
 *           the timeout to a smaller value
 *       <li>If the timeout is exceeded, the implementation MUST throw TimeoutException, with the
 *           timeout that was applied to the subprocess (TODO)
 *     </ul>
 * </ul>
 *
 * <h2>Optimistic Concurrency</h2>
 * Bazel may choose to execute a spawn using multiple {@link SpawnRunner} implementations
 * simultaneously in order to minimize total latency. This is especially useful for builds with few
 * actions where remotely executing the actions incurs high round trip times.
 * <ul>
 *   <li>All implementations MUST call {@link SpawnExecutionPolicy#lockOutputFiles} before writing
 *       to any of the output files, but may write to stdout and stderr without calling it. Instead,
 *       all callers must provide temporary locations for stdout & stderr if they ever call multiple
 *       {@link SpawnRunner} implementations concurrently. Spawn runners that use the local machine
 *       MUST either call it before starting the subprocess, or ensure that subprocesses write to
 *       temporary locations (for example by running in a mount namespace) and then copy or move the
 *       outputs into place.
 *   <li>Implementations SHOULD delay calling {@link SpawnExecutionPolicy#lockOutputFiles} until
 *       just before writing.
 * </ul>
 */
public interface SpawnRunner {
  /**
   * A helper class to provide additional tools and methods to {@link SpawnRunner} implementations.
   *
   * <p>This interface may change without notice.
   */
  public interface SpawnExecutionPolicy {
    /**
     * Returns whether inputs should be prefetched to the local machine using {@link
     * ActionInputPrefetcher} if the spawn is executed locally (with or without sandboxing).
     */
    // TODO(ulfjack): Use an execution info value instead.
    boolean shouldPrefetchInputsForLocalExecution(Spawn spawn);

    /**
     * The input file cache for this specific spawn.
     */
    ActionInputFileCache getActionInputFileCache();

    /**
     * All implementations must call this method before writing to the provided stdout / stderr or
     * to any of the output file locations. This method is used to coordinate - implementations
     * must throw an {@link InterruptedException} for all but one caller.
     */
    void lockOutputFiles() throws InterruptedException;

    /**
     * Returns the timeout that should be applied for the given {@link Spawn} instance.
     */
    long getTimeoutMillis();

    /**
     * The files to which to write stdout and stderr.
     */
    FileOutErr getFileOutErr();

    SortedMap<PathFragment, ActionInput> getInputMapping() throws IOException;
  }

  /**
   * Run the given spawn.
   *
   * @param spawn the spawn to run
   * @param policy a helper that provides additional parameters
   * @return the result from running the spawn
   * @throws InterruptedException if the calling thread was interrupted, or if the runner could not
   *         lock the output files (see {@link SpawnExecutionPolicy#lockOutputFiles()})
   * @throws IOException if something went wrong reading or writing to the local file system
   * @throws ExecException if the request is malformed
   */
  SpawnResult exec(
      Spawn spawn,
      SpawnExecutionPolicy policy)
          throws InterruptedException, IOException, ExecException;
}
