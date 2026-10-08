// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.util.CommandBuilder;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.Map;
import java.util.UUID;

/** Utility for asynchronously deleting directories via background daemonized processes. */
public final class AsyncDirectoryCleaner {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private AsyncDirectoryCleaner() {}

  /**
   * Atomically renames {@code path} to a temporary name in the same parent directory.
   *
   * <p>Keeping the temporary directory in the same parent directory ensures it remains in the same
   * filesystem, allowing the rename to be atomic and instantaneous.
   *
   * @param path the directory to move
   * @return the temporary {@link Path}
   * @throws IOException if {@code path} has no parent directory or the rename fails
   */
  public static Path moveToTempDirectory(Path path) throws IOException {
    Path parentDir = path.getParentDirectory();
    if (parentDir == null) {
      throw new IOException("Cannot move root directory " + path + " to temporary directory");
    }
    String tempBaseName =
        path.getBaseName() + "_tmp_" + ProcessHandle.current().pid() + "_" + UUID.randomUUID();
    Path tempPath = parentDir.getChild(tempBaseName);
    path.renameTo(tempPath);
    return tempPath;
  }

  /**
   * Constructs the {@link Command} used to daemonize directory deletion.
   *
   * <p>The executed shell script recursively makes all directories writable ({@code find ./$1 -type
   * d -not -perm -u=rwx -exec chmod -f u=rwx {} +}) and deletes the directory ({@code rm -rf
   * ./$1}). Passing {@code dirNameToDelete} as positional argument {@code $1} safely avoids word
   * splitting, globbing, or shell quotation issues.
   */
  private static Command createDaemonizedDeleteCommand(
      Path daemonize, Path parentDir, String dirNameToDelete, Map<String, String> clientEnv) {
    return new CommandBuilder(clientEnv)
        .addArg(daemonize.getPathString())
        .addArgs("-l", "/dev/null")
        .addArgs("-p", "/dev/null")
        .addArg("--")
        // daemonize takes <path> <arg0> <arg1>... The first "/bin/sh" is consumed by
        // daemonize as argv[0] of the executed shell.
        // We pass dirNameToDelete safely as a positional argument ($1) so it is never
        // subject to word splitting, globbing, or shell evaluation. In POSIX shell,
        // parameter expansion inside double quotes ("./$1") treats quotes inside $1
        // as literal characters (no quote re-parsing), ensuring safe execution even if
        // the directory name contains whitespace, single/double quotes, or special
        // characters.
        .addArgs(
            "/bin/sh",
            "/bin/sh",
            "-c",
            "/usr/bin/find \"./$1\" -type d -not -perm -u=rwx -exec /bin/chmod -f u=rwx {}"
                + " +; /bin/rm -rf \"./$1\"",
            "_",
            dirNameToDelete)
        .setWorkingDir(parentDir)
        .build();
  }

  /**
   * Spawns a background daemon process using {@code daemonize} to delete {@code dirNameToDelete}
   * inside {@code parentDir}.
   *
   * @param daemonize path to the daemonize binary
   * @param parentDir working directory containing {@code dirNameToDelete}
   * @param dirNameToDelete relative name of the directory inside {@code parentDir} to delete
   * @param clientEnv environment variables for the daemon process
   * @return the {@link CommandResult} of spawning daemonize
   * @throws CommandException if spawning fails
   * @throws InterruptedException if interrupted while spawning
   */
  @CanIgnoreReturnValue
  public static CommandResult spawnDaemonizedDeletion(
      Path daemonize, Path parentDir, String dirNameToDelete, Map<String, String> clientEnv)
      throws CommandException, InterruptedException {
    logger.atInfo().log("Spawning daemonized deletion for %s in %s", dirNameToDelete, parentDir);
    CommandResult result =
        createDaemonizedDeleteCommand(daemonize, parentDir, dirNameToDelete, clientEnv).execute();
    logger.atInfo().log("Daemonized deletion spawn status: %s", result.terminationStatus());
    return result;
  }
}
