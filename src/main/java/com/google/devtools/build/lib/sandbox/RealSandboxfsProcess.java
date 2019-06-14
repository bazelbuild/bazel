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

import static com.google.common.base.Preconditions.checkNotNull;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.shell.SubprocessBuilder.StreamAction;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** A sandboxfs implementation that uses an external sandboxfs binary to manage the mount point. */
final class RealSandboxfsProcess implements SandboxfsProcess {
  private static final Logger log = Logger.getLogger(RealSandboxfsProcess.class.getName());

  /** Directory on which the sandboxfs is serving. */
  private final Path mountPoint;

  /**
   * Process handle to the sandboxfs instance.  Null only after {@link #destroy()} has been invoked.
   */
  private @Nullable Subprocess process;

  /**
   * Writer with which to send data to the sandboxfs instance.  Null only after {@link #destroy()}
   * has been invoked.
   */
  private @Nullable BufferedWriter processStdIn;

  /**
   * Reader with which to receive data from the sandboxfs instance.  Null only after
   * {@link #destroy()} has been invoked.
   */
  private @Nullable BufferedReader processStdOut;

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
  private RealSandboxfsProcess(Path mountPoint, Subprocess process) {
    this.mountPoint = mountPoint;

    this.process = process;
    this.processStdIn = new BufferedWriter(
        new OutputStreamWriter(process.getOutputStream(), UTF_8));
    this.processStdOut = new BufferedReader(
        new InputStreamReader(process.getInputStream(), UTF_8));

    this.shutdownHook =
        new Thread(
            () -> {
              try {
                this.destroy();
              } catch (Exception e) {
                log.warning("Failed to destroy running sandboxfs instance; mount point may have "
                    + "been left behind: " + e);
              }
            });
    Runtime.getRuntime().addShutdownHook(shutdownHook);
  }

  /**
   * Checks if the given sandboxfs binary is available and is valid.
   *
   * @param binary path to the sandboxfs binary that will later be used in the {@link #mount} call.
   * @return true if the binary looks good, false otherwise
   * @throws IOException if there is a problem trying to start the subprocess
   */
  static boolean isAvailable(PathFragment binary) {
    Subprocess process;
    try {
      process =
          new SubprocessBuilder()
              .setArgv(binary.getPathString(), "--version")
              .setStdout(StreamAction.STREAM)
              .redirectErrorStream(true)
              .start();
    } catch (IOException e) {
      log.warning("sandboxfs binary at " + binary + " seems to be missing; got error " + e);
      return false;
    }

    ByteArrayOutputStream outErrBytes = new ByteArrayOutputStream();
    try {
      ByteStreams.copy(process.getInputStream(), outErrBytes);
    } catch (IOException e) {
      try {
        outErrBytes.write(("Failed to read stdout: " + e).getBytes());
      } catch (IOException e2) {
        // Should not really have happened. There is nothing we can do.
      }
    }
    String outErr = outErrBytes.toString().replaceFirst("\n$", "");

    int exitCode = waitForProcess(process);
    if (exitCode == 0) {
      // TODO(jmmv): Validate the version number and ensure we support it. Would be nice to reuse
      // the DottedVersion logic from the Apple rules.
      if (outErr.matches("sandboxfs .*")) {
        return true;
      } else {
        log.warning(
            "sandboxfs binary at "
                + binary
                + " exists but doesn't seem valid; output was: "
                + outErr);
        return false;
      }
    } else {
      log.warning(
          "sandboxfs binary at "
              + binary
              + " returned non-zero exit code "
              + exitCode
              + " and output "
              + outErr);
      return false;
    }
  }

  /**
   * Mounts a new sandboxfs instance.
   *
   * <p>The root of the file system instance is left unmapped which means that it remains as
   * read-only throughout the lifetime of this instance.  Writable subdirectories can later be
   * mapped via {@link #map(List)}.
   *
   * @param binary path to the sandboxfs binary.  This is a {@link PathFragment} and not a
   *     {@link Path} because we want to support "bare" (non-absolute) names for the location of
   *     the sandboxfs binary; such names are automatically looked for in the {@code PATH}.
   * @param mountPoint directory on which to mount the sandboxfs instance
   * @param logFile path to the file that will receive all sandboxfs logging output
   * @return a new handle that represents the running process
   * @throws IOException if there is a problem starting the process
   */
  static SandboxfsProcess mount(PathFragment binary, Path mountPoint, Path logFile)
      throws IOException {
    log.info("Mounting sandboxfs (" + binary + ") onto " + mountPoint);

    // TODO(jmmv): Before starting a sandboxfs serving instance, we must query the current version
    // of sandboxfs and check if we support its communication protocol.

    ImmutableList.Builder<String> argvBuilder = ImmutableList.builder();

    argvBuilder.add(binary.getPathString());

    // On macOS, we need to allow users other than self to access the sandboxfs instance.  This is
    // necessary because macOS's amfid, which runs as root, has to have access to the binaries
    // within the sandbox in order to validate signatures. See:
    // http://julio.meroh.net/2017/10/fighting-execs-sandboxfs-macos.html
    argvBuilder.add(OS.getCurrent() == OS.DARWIN ? "--allow=other" : "--allow=self");

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
    RealSandboxfsProcess sandboxfs = new RealSandboxfsProcess(mountPoint, process);
    // TODO(jmmv): We should have a better mechanism to wait for sandboxfs to start successfully but
    // sandboxfs currently provides no interface to do so.  Just try to push an empty configuration
    // and see if it works.
    try {
      sandboxfs.reconfigure("[]\n\n");
    } catch (IOException e) {
      destroyProcess(process);
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

  /**
   * Waits for a process to terminate.
   *
   * @param process the process to wait for
   * @return the exit code of the terminated process
   */
  private static int waitForProcess(Subprocess process) {
    boolean interrupted = false;
    try {
      while (true) {
        try {
          process.waitFor();
          break;
        } catch (InterruptedException ie) {
          interrupted = true;
        }
      }
    } finally {
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }
    return process.exitValue();
  }

  /**
   * Destroys a process and waits for it to exit.
   *
   * @param process the process to destroy
   */
  // TODO(jmmv): This is adapted from Worker.java. Should probably replace both with a new variant
  // of Uninterruptibles.callUninterruptibly that takes a lambda instead of a callable.
  private static void destroyProcess(Subprocess process) {
    process.destroy();
    waitForProcess(process);
  }

  @Override
  public synchronized void destroy() {
    if (shutdownHook != null) {
      Runtime.getRuntime().removeShutdownHook(shutdownHook);
      shutdownHook = null;
    }

    if (processStdIn != null) {
      try {
        processStdIn.close();
      } catch (IOException e) {
        log.warning("Failed to close sandboxfs's stdin pipe: " + e);
      }
      processStdIn = null;
    }

    if (processStdOut != null) {
      try {
        processStdOut.close();
      } catch (IOException e) {
        log.warning("Failed to close sandboxfs's stdout pipe: " + e);
      }
      processStdOut = null;
    }

    if (process != null) {
      destroyProcess(process);
      process = null;
    }
  }

  /**
   * Pushes a new configuration to sandboxfs and waits for acceptance.
   *
   * @param config the configuration chunk to push to sandboxfs
   * @throws IOException if sandboxfs cannot be reconfigured either because of an error in the
   *     configuration or because we failed to communicate with the subprocess
   */
  private synchronized void reconfigure(String config) throws IOException {
    checkNotNull(processStdIn, "sandboxfs already has been destroyed");
    processStdIn.write(config);
    processStdIn.flush();

    checkNotNull(processStdOut, "sandboxfs has already been destroyed");
    String done = processStdOut.readLine();
    if (done == null) {
      throw new IOException("premature end of output from sandboxfs");
    }
    if (!done.equals("Done")) {
      throw new IOException("received unknown string from sandboxfs: " + done + "; expected Done");
    }
  }

  @Override
  public void map(List<Mapping> mappings) throws IOException {
    Function<Mapping, String> formatMapping =
        (mapping) -> String.format(
            "{\"Map\": {\"Mapping\": \"%s\", \"Target\": \"%s\", \"Writable\": %s}}",
            mapping.path(), mapping.target(), mapping.writable() ? "true" : "false");

    StringBuilder sb = new StringBuilder();
    sb.append("[\n");
    sb.append(mappings.stream().map(formatMapping).collect(Collectors.joining(",\n")));
    sb.append("]\n\n");
    reconfigure(sb.toString());
  }

  @Override
  public void unmap(PathFragment mapping) throws IOException {
    reconfigure(String.format("[{\"Unmap\": \"%s\"}]\n\n", mapping));
  }
}
