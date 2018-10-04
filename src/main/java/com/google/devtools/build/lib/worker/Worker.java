// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;

/**
 * Interface to a worker process running as a child process.
 *
 * <p>A worker process must follow this protocol to be usable via this class: The worker process is
 * spawned on demand. The worker process is free to exit whenever necessary, as new instances will
 * be relaunched automatically. Communication happens via the WorkerProtocol protobuf, sent to and
 * received from the worker process via stdin / stdout.
 *
 * <p>Other code in Blaze can talk to the worker process via input / output streams provided by this
 * class.
 */
class Worker {
  private final WorkerKey workerKey;
  private final int workerId;
  private final Path workDir;
  private final Path logFile;

  private Subprocess process;
  private Thread shutdownHook;

  Worker(WorkerKey workerKey, int workerId, final Path workDir, Path logFile) {
    this.workerKey = workerKey;
    this.workerId = workerId;
    this.workDir = workDir;
    this.logFile = logFile;

    final Worker self = this;
    this.shutdownHook =
        new Thread(
            () -> {
              try {
                self.shutdownHook = null;
                self.destroy();
              } catch (IOException e) {
                // We can't do anything here.
              }
            });
    Runtime.getRuntime().addShutdownHook(shutdownHook);
  }

  void createProcess() throws IOException {
    List<String> args = workerKey.getArgs();
    File executable = new File(args.get(0));
    if (!executable.isAbsolute() && executable.getParent() != null) {
      args = new ArrayList<>(args);
      args.set(0, new File(workDir.getPathFile(), args.get(0)).getAbsolutePath());
    }
    SubprocessBuilder processBuilder = new SubprocessBuilder();
    processBuilder.setArgv(args);
    processBuilder.setWorkingDirectory(workDir.getPathFile());
    processBuilder.setStderr(logFile.getPathFile());
    processBuilder.setEnv(workerKey.getEnv());
    this.process = processBuilder.start();
  }

  void destroy() throws IOException {
    if (shutdownHook != null) {
      Runtime.getRuntime().removeShutdownHook(shutdownHook);
    }
    if (process != null) {
      destroyProcess(process);
    }
  }

  /**
   * Destroys a process and waits for it to exit. This is necessary for the child to not become a
   * zombie.
   *
   * @param process the process to destroy.
   */
  private static void destroyProcess(Subprocess process) {
    boolean wasInterrupted = false;
    try {
      process.destroy();
      while (true) {
        try {
          process.waitFor();
          return;
        } catch (InterruptedException ie) {
          wasInterrupted = true;
        }
      }
    } finally {
      // Read this for detailed explanation: http://www.ibm.com/developerworks/library/j-jtp05236/
      if (wasInterrupted) {
        Thread.currentThread().interrupt(); // preserve interrupted status
      }
    }
  }

  /**
   * Returns a unique id for this worker. This is used to distinguish different worker processes in
   * logs and messages.
   */
  int getWorkerId() {
    return this.workerId;
  }

  HashCode getWorkerFilesCombinedHash() {
    return workerKey.getWorkerFilesCombinedHash();
  }

  SortedMap<PathFragment, HashCode> getWorkerFilesWithHashes() {
    return workerKey.getWorkerFilesWithHashes();
  }

  boolean isAlive() {
    // This is horrible, but Process.isAlive() is only available from Java 8 on and this is the
    // best we can do prior to that.
    return !process.finished();
  }

  InputStream getInputStream() {
    return process.getInputStream();
  }

  OutputStream getOutputStream() {
    return process.getOutputStream();
  }

  public void prepareExecution(
      Map<PathFragment, Path> inputFiles, SandboxOutputs outputs, Set<PathFragment> workerFiles)
      throws IOException {
    if (process == null) {
      createProcess();
    }
  }

  public void finishExecution(Path execRoot) throws IOException {}

  public Path getLogFile() {
    return logFile;
  }
}
