// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.versioning.GnuVersionParser;
import com.google.devtools.build.lib.versioning.ParseException;
import com.google.devtools.build.lib.versioning.SemVer;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * A sandboxfs implementation that uses an external sandboxfs binary to manage the mount point.
 *
 * <p>This class implements common code to generalize the interactions with sandboxfs, but delegates
 * to its subclassess once the version of sandboxfs in use has been determined. The subclasses
 * implement logic specific to each version to provide compatibility with the different versions of
 * sandboxfs that the user might have installed.
 */
abstract class RealSandboxfsProcess implements SandboxfsProcess {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * Contains the {@code --allow} flag to pass to sandboxfs.
   *
   * <p>On macOS, we need to allow users other than self to access the sandboxfs instance. This is
   * necessary because macOS's amfid, which runs as root, has to have access to the binaries within
   * the sandbox in order to validate signatures. See:
   * http://jmmv.dev/2017/10/fighting-execs-sandboxfs-macos.html
   */
  @VisibleForTesting
  static final String ALLOW_FLAG = OS.getCurrent() == OS.DARWIN ? "--allow=other" : "--allow=self";

  /** Directory on which the sandboxfs is serving. */
  private final Path mountPoint;

  /**
   * Process handle to the sandboxfs instance.  Null only after {@link #destroy()} has been invoked.
   */
  private @Nullable Subprocess process;

  /**
   * Shutdown hook to stop the sandboxfs instance on abrupt termination.  Null only after
   * {@link #destroy()} has been invoked.
   */
  private @Nullable Thread shutdownHook;

  /**
   * Initializes a new sandboxfs process instance.
   *
   * @param process process handle for the already-running sandboxfs instance
   */
  RealSandboxfsProcess(Path mountPoint, Subprocess process) {
    this.mountPoint = mountPoint;
    this.process = process;

    this.shutdownHook =
        new Thread(
            () -> {
              try {
                this.destroy();
              } catch (Exception e) {
                logger.atWarning().withCause(e).log(
                    "Failed to destroy running sandboxfs instance; mount point may have "
                        + "been left behind");
              }
            });
    Runtime.getRuntime().addShutdownHook(shutdownHook);
  }

  /**
   * Mounts a new sandboxfs instance.
   *
   * <p>The root of the file system instance is left unmapped which means that it remains as
   * read-only throughout the lifetime of this instance. Writable subdirectories can later be mapped
   * via {@link #createSandbox}.
   *
   * @param binary path to the sandboxfs binary. This is a {@link PathFragment} and not a {@link
   *     Path} because we want to support "bare" (non-absolute) names for the location of the
   *     sandboxfs binary; such names are automatically looked for in the {@code PATH}.
   * @param mountPoint directory on which to mount the sandboxfs instance
   * @param logFile path to the file that will receive all sandboxfs logging output
   * @return a new handle that represents the running process
   * @throws IOException if there is a problem starting the process
   */
  static SandboxfsProcess mount(PathFragment binary, Path mountPoint, Path logFile)
      throws IOException {
    logger.atInfo().log("Mounting sandboxfs (%s) onto %s", binary, mountPoint);

    GnuVersionParser<SemVer> parser = new GnuVersionParser<>("sandboxfs", SemVer::parse);
    SemVer version;
    try {
      version = parser.fromProgram(binary);
    } catch (IOException | ParseException e) {
      throw new IOException("Failed to get sandboxfs version from " + binary, e);
    }

    ImmutableList.Builder<String> argvBuilder = ImmutableList.builder();
    argvBuilder.add(binary.getPathString());
    argvBuilder.add(ALLOW_FLAG);

    // TODO(jmmv): Pass flags to enable sandboxfs' debugging support (--listen_address and --debug)
    // when requested by the user via --sandbox_debug.  Tricky because we have to figure out how to
    // deal with port numbers (which sandboxfs can autoassign, but doesn't currently promise a way
    // to tell us back what it picked).

    argvBuilder.add(mountPoint.getPathString());

    SubprocessBuilder processBuilder = new SubprocessBuilder();
    processBuilder.setArgv(argvBuilder.build());
    processBuilder.setStderr(logFile.getPathFile());
    processBuilder.setEnv(ImmutableMap.of(
        // sandboxfs may need to locate fusermount depending on the FUSE implementation so pass the
        // PATH to the subprocess (which we assume is sufficient).
        "PATH", System.getenv("PATH")));

    Subprocess process = processBuilder.start();
    RealSandboxfsProcess sandboxfs;
    if (version.compareTo(SemVer.from(0, 2)) >= 0) {
      sandboxfs = new RealSandboxfs02Process(mountPoint, process);
    } else {
      sandboxfs = new RealSandboxfs01Process(mountPoint, process);
    }
    try {
      // Create an empty sandbox to ensure sandboxfs is successfully serving.
      sandboxfs.createSandbox("empty", (mapper) -> {});
    } catch (IOException e) {
      process.destroyAndWait();
      throw new IOException("sandboxfs failed to start", e);
    }
    return sandboxfs;
  }

  @Override
  public Path getMountPoint() {
    return mountPoint;
  }

  @Override
  public boolean isAlive() {
    return process != null && !process.finished();
  }

  @Override
  public synchronized void destroy() {
    if (shutdownHook != null) {
      Runtime.getRuntime().removeShutdownHook(shutdownHook);
      shutdownHook = null;
    }

    if (process != null) {
      try {
        process.getOutputStream().close();
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("Failed to close sandboxfs's stdin pipe");
      }

      try {
        process.getInputStream().close();
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("Failed to close sandboxfs's stdout pipe");
      }

      process.destroyAndWait();
      process = null;
    }
  }
}
