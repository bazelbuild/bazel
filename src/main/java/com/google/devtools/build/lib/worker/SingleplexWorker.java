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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.sandbox.CgroupsInfo;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.sandbox.cgroups.VirtualCgroupFactory;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.File;
import java.io.IOException;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Interface to a worker process running as a single child process.
 *
 * <p>A worker process must follow this protocol to be usable via this class: The worker process is
 * spawned on demand. The worker process is free to exit whenever necessary, as new instances will
 * be relaunched automatically. Communication happens via the WorkerProtocol protobuf, sent to and
 * received from the worker process via stdin / stdout.
 *
 * <p>Other code in Blaze can talk to the worker process via input / output streams provided by this
 * class.
 */
class SingleplexWorker extends Worker {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** The execution root of the worker. */
  protected final Path workDir;

  /**
   * Stream for recording the WorkResponse as it's read, so that it can be printed in the case of
   * parsing failures.
   */
  @Nullable private RecordingInputStream recordingInputStream;

  /** The implementation of the worker protocol (JSON or Proto). */
  @Nullable private WorkerProtocolImpl workerProtocol;

  private Subprocess process;

  /** True if we deliberately destroyed this process. */
  private boolean wasDestroyed;

  /**
   * Shutdown hook to make sure we wait for the process to finish on JVM shutdown, to avoid creating
   * zombie processes. Unfortunately, shutdown hooks are not guaranteed to be called, but this is
   * the best we can do. This must be set when a process is created.
   */
  protected Thread shutdownHook;

  protected WorkerOptions options;
  protected final VirtualCgroupFactory cgroupFactory;

  SingleplexWorker(
      WorkerKey workerKey,
      int workerId,
      final Path workDir,
      Path logFile,
      WorkerOptions options,
      @Nullable VirtualCgroupFactory cgroupFactory) {
    super(workerKey, workerId, logFile, new WorkerProcessStatus());
    this.workDir = workDir;
    this.options = options;
    this.cgroupFactory = cgroupFactory;
  }

  protected Subprocess createProcess(ImmutableMap<String, String> clientEnv)
      throws IOException, UserExecException {
    ImmutableList<String> args = makeExecPathAbsolute(workerKey.getArgs());
    Subprocess process = createProcessBuilder(args, clientEnv).start();
    if (cgroupFactory != null) {
      cgroup = cgroupFactory.create(workerId, ImmutableMap.of());
    } else if (options.useCgroupsOnLinux && CgroupsInfo.isSupported()) {
      cgroup =
          CgroupsInfo.getBlazeSpawnsCgroup()
              .createIndividualSpawnCgroup(
                  /* dirName= */ "worker_" + workerId, /* memoryLimitMb= */ 0);
    }
    if (cgroup != null && cgroup.exists()) {
      cgroup.addProcess(process.getProcessId());
    }
    return process;
  }

  protected SubprocessBuilder createProcessBuilder(
      ImmutableList<String> argv, ImmutableMap<String, String> clientEnv) {
    SubprocessBuilder processBuilder = new SubprocessBuilder(clientEnv);
    processBuilder.setArgv(argv);
    processBuilder.setWorkingDirectory(workDir.getPathFile());
    processBuilder.setStderr(logFile.getPathFile());
    processBuilder.setEnv(workerKey.getEnv());
    return processBuilder;
  }

  @Override
  public boolean isSandboxed() {
    return false;
  }

  @Override
  public void prepareExecution(
      SandboxInputs inputFiles,
      SandboxOutputs outputs,
      Set<PathFragment> workerFiles,
      ImmutableMap<String, String> clientEnv)
      throws IOException, InterruptedException, UserExecException {
    if (process == null) {
      addShutdownHook();
      process = createProcess(clientEnv);
      logger.atInfo().log(
          "Created worker process %s for worker id %d", process.getProcessId(), workerId);
      status.maybeUpdateStatus(WorkerProcessStatus.Status.ALIVE);
      recordingInputStream = new RecordingInputStream(process.getInputStream());
    }
    if (workerProtocol == null) {
      switch (workerKey.getProtocolFormat()) {
        case JSON:
          workerProtocol = new JsonWorkerProtocol(process.getOutputStream(), recordingInputStream);
          break;
        case PROTO:
          workerProtocol = new ProtoWorkerProtocol(process.getOutputStream(), recordingInputStream);
          break;
      }
    }
  }

  void addShutdownHook() {
    this.shutdownHook =
        new Thread(
            () -> {
              this.shutdownHook = null;
              this.destroy();
            });
    Runtime.getRuntime().addShutdownHook(shutdownHook);
  }

  /**
   * Makes sure that the executable (first element of argument list) is an absolute path. Necessary
   * on Windows (https://github.com/bazelbuild/bazel/commit/8efc3ef0)
   */
  protected ImmutableList<String> makeExecPathAbsolute(ImmutableList<String> args) {
    File executable = new File(args.get(0));
    if (!executable.isAbsolute() && executable.getParent() != null) {
      return ImmutableList.<String>builderWithExpectedSize(args.size())
          .add(new File(workDir.getPathFile(), args.get(0)).getAbsolutePath())
          .addAll(args.subList(1, args.size()))
          .build();
    } else {
      return args;
    }
  }

  @Override
  void putRequest(WorkRequest request) throws IOException {
    workerProtocol.putRequest(request);
  }

  @Override
  WorkResponse getResponse(int requestId) throws IOException, InterruptedException {
    recordingInputStream.startRecording(4096);
    while (recordingInputStream.available() == 0) {
      Thread.sleep(10);
      if (!process.isAlive()) {
        throw new IOException(
            String.format(
                "Worker process for %s died while waiting for response", workerKey.getMnemonic()));
      }
    }
    return workerProtocol.getResponse();
  }

  @Override
  void destroy() {
    if (workerProtocol != null) {
      try {
        workerProtocol.close();
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("Caught IOException while closing worker protocol.");
      }
      workerProtocol = null;
    }
    if (shutdownHook != null) {
      try {
        Runtime.getRuntime().removeShutdownHook(shutdownHook);
      } catch (IllegalStateException e) {
        // Can only happen if we're already in shutdown, in which case we don't care.
      }
    }
    if (process != null) {
      wasDestroyed = true;
      process.destroyAndWait();
    }
    if (cgroupFactory != null) {
      cgroupFactory.remove(workerId);
    }
    status.setKilled();
  }

  /** Returns true if this process is dead but we didn't deliberately kill it. */
  @Override
  boolean diedUnexpectedly() {
    return process != null && !wasDestroyed && !process.isAlive();
  }

  @Override
  public Optional<Integer> getExitValue() {
    return process != null && !process.isAlive()
        ? Optional.of(process.exitValue())
        : Optional.empty();
  }

  @Override
  String getRecordingStreamMessage() {
    recordingInputStream.readRemaining();
    return recordingInputStream.getRecordedDataAsString();
  }

  @Override
  public String toString() {
    return workerKey.getMnemonic() + " worker #" + workerId;
  }

  @Override
  public long getProcessId() {
    if (process == null) {
      return -1;
    }

    return process.getProcessId();
  }
}
