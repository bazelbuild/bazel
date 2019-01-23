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

package com.google.devtools.build.remote.worker;

import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static java.util.logging.Level.FINE;
import static java.util.logging.Level.INFO;
import static java.util.logging.Level.SEVERE;
import static java.util.logging.Level.WARNING;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Command.EnvironmentVariable;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ExecutionGrpc.ExecutionImplBase;
import build.bazel.remote.execution.v2.Platform;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.WaitExecutionRequest;
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.remote.CacheNotFoundException;
import com.google.devtools.build.lib.remote.ExecutionStatusException;
import com.google.devtools.build.lib.remote.SimpleBlobStoreActionCache;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.shell.FutureCommandResult;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.longrunning.Operation;
import com.google.protobuf.Any;
import com.google.protobuf.util.Durations;
import com.google.rpc.Code;
import com.google.rpc.Status;
import io.grpc.Context;
import io.grpc.StatusException;
import io.grpc.protobuf.StatusProto;
import io.grpc.stub.StreamObserver;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/** A basic implementation of an {@link ExecutionImplBase} service. */
final class ExecutionServer extends ExecutionImplBase {
  private static final Logger logger = Logger.getLogger(ExecutionServer.class.getName());

  // The name of the container image entry in the Platform proto
  // (see third_party/googleapis/devtools/remoteexecution/*/remote_execution.proto and
  // experimental_remote_platform_override in
  // src/main/java/com/google/devtools/build/lib/remote/RemoteOptions.java)
  private static final String CONTAINER_IMAGE_ENTRY_NAME = "container-image";
  private static final String DOCKER_IMAGE_PREFIX = "docker://";

  // How long to wait for the uid command.
  private static final Duration uidTimeout = Duration.ofMillis(30);

  private static final int LOCAL_EXEC_ERROR = -1;

  private final Path workPath;
  private final Path sandboxPath;
  private final RemoteWorkerOptions workerOptions;
  private final SimpleBlobStoreActionCache cache;
  private final ConcurrentHashMap<String, ListenableFuture<ActionResult>> operationsCache;
  private final ListeningExecutorService executorService;
  private final DigestUtil digestUtil;

  public ExecutionServer(
      Path workPath,
      Path sandboxPath,
      RemoteWorkerOptions workerOptions,
      SimpleBlobStoreActionCache cache,
      ConcurrentHashMap<String, ListenableFuture<ActionResult>> operationsCache,
      DigestUtil digestUtil) {
    this.workPath = workPath;
    this.sandboxPath = sandboxPath;
    this.workerOptions = workerOptions;
    this.cache = cache;
    this.operationsCache = operationsCache;
    this.digestUtil = digestUtil;
    ThreadPoolExecutor realExecutor =
        new ThreadPoolExecutor(
            // This is actually the max number of concurrent jobs.
            workerOptions.jobs,
            // Since we use an unbounded queue, the executor ignores this value, but it still checks
            // that it is greater or equal to the value above.
            workerOptions.jobs,
            // Shut down idle threads after one minute. Threads aren't all that expensive, but we
            // also
            // don't need to keep them around if we don't need them.
            1,
            TimeUnit.MINUTES,
            // We use an unbounded queue for now.
            // TODO(ulfjack): We need to reject work eventually.
            new LinkedBlockingQueue<>(),
            new ThreadFactoryBuilder().setNameFormat("subprocess-handler-%d").build());
    // Allow the core threads to die.
    realExecutor.allowCoreThreadTimeOut(true);
    this.executorService = MoreExecutors.listeningDecorator(realExecutor);
  }

  @Override
  public void waitExecution(WaitExecutionRequest wr, StreamObserver<Operation> responseObserver) {
    final String opName = wr.getName();
    ListenableFuture<ActionResult> future = operationsCache.get(opName);
    if (future == null) {
      responseObserver.onError(
          StatusProto.toStatusRuntimeException(
              Status.newBuilder()
                  .setCode(Code.NOT_FOUND.getNumber())
                  .setMessage("Operation not found: " + opName)
                  .build()));
      return;
    }
    waitExecution(opName, future, responseObserver);
  }

  private void waitExecution(
      String opName,
      ListenableFuture<ActionResult> future,
      StreamObserver<Operation> responseObserver) {
    future.addListener(
        () -> {
          try {
            try {
              ActionResult result = future.get();
              responseObserver.onNext(
                  Operation.newBuilder()
                      .setName(opName)
                      .setDone(true)
                      .setResponse(Any.pack(ExecuteResponse.newBuilder().setResult(result).build()))
                      .build());
              responseObserver.onCompleted();
            } catch (ExecutionException e) {
              Throwables.throwIfUnchecked(e.getCause());
              throw (Exception) e.getCause();
            }
          } catch (Exception e) {
            ExecuteResponse resp;
            if (e instanceof ExecutionStatusException) {
              resp = ((ExecutionStatusException) e).getResponse();
            } else {
              logger.log(Level.SEVERE, "Work failed: " + opName, e);
              resp =
                  ExecuteResponse.newBuilder()
                      .setStatus(StatusUtils.internalErrorStatus(e))
                      .build();
            }
            responseObserver.onNext(
                Operation.newBuilder()
                    .setName(opName)
                    .setDone(true)
                    .setResponse(Any.pack(resp))
                    .build());
            responseObserver.onCompleted();
            if (e instanceof InterruptedException) {
              Thread.currentThread().interrupt();
            }
          } finally {
            operationsCache.remove(opName);
          }
        },
        MoreExecutors.directExecutor());
  }

  @Override
  public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
    final String opName = UUID.randomUUID().toString();
    ListenableFuture<ActionResult> future =
        executorService.submit(Context.current().wrap(() -> execute(request, opName)));
    operationsCache.put(opName, future);
    // Send the first operation.
    responseObserver.onNext(Operation.newBuilder().setName(opName).build());
    // When the operation completes, send the result.
    waitExecution(opName, future, responseObserver);
  }

  private ActionResult execute(ExecuteRequest request, String id)
      throws IOException, InterruptedException, StatusException {
    Path tempRoot = workPath.getRelative("build-" + id);
    String workDetails = "";
    try {
      tempRoot.createDirectory();
      RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
      workDetails =
          String.format(
              "build-request-id: %s command-id: %s action-id: %s",
              meta.getCorrelatedInvocationsId(), meta.getToolInvocationId(), meta.getActionId());
      logger.log(FINE, "Received work for: {0}", workDetails);
      ActionResult result = execute(request.getActionDigest(), tempRoot);
      logger.log(FINE, "Completed {0}.", workDetails);
      return result;
    } catch (Exception e) {
      logger.log(Level.SEVERE, "Work failed: {0} {1}.", new Object[] {workDetails, e});
      throw e;
    } finally {
      if (workerOptions.debug) {
        logger.log(INFO, "Preserving work directory {0}.", tempRoot);
      } else {
        try {
          FileSystemUtils.deleteTree(tempRoot);
        } catch (IOException e) {
          logger.log(
              SEVERE,
              String.format(
                  "Failed to delete tmp directory %s: %s",
                  tempRoot, Throwables.getStackTraceAsString(e)));
        }
      }
    }
  }

  private ActionResult execute(Digest actionDigest, Path execRoot)
      throws IOException, InterruptedException, StatusException {
    Command command = null;
    Action action = null;
    ActionKey actionKey = digestUtil.asActionKey(actionDigest);
    try {
      action = Action.parseFrom(getFromFuture(cache.downloadBlob(actionDigest)));
      command = Command.parseFrom(getFromFuture(cache.downloadBlob(action.getCommandDigest())));
      cache.downloadTree(action.getInputRootDigest(), execRoot);
    } catch (CacheNotFoundException e) {
      throw StatusUtils.notFoundError(e.getMissingDigest());
    }

    List<Path> outputs = new ArrayList<>(command.getOutputFilesList().size());
    for (String output : command.getOutputFilesList()) {
      Path file = execRoot.getRelative(output);
      if (file.exists()) {
        throw new FileAlreadyExistsException("Output file already exists: " + file);
      }
      FileSystemUtils.createDirectoryAndParents(file.getParentDirectory());
      outputs.add(file);
    }
    for (String output : command.getOutputDirectoriesList()) {
      Path file = execRoot.getRelative(output);
      if (file.exists()) {
        throw new FileAlreadyExistsException("Output directory/file already exists: " + file);
      }
      FileSystemUtils.createDirectoryAndParents(file.getParentDirectory());
      outputs.add(file);
    }

    // TODO(ulfjack): This is basically a copy of LocalSpawnRunner. Ideally, we'd use that
    // implementation instead of copying it.
    com.google.devtools.build.lib.shell.Command cmd = getCommand(command, execRoot.getPathString());
    long startTime = System.currentTimeMillis();
    CommandResult cmdResult = null;

    FutureCommandResult futureCmdResult = null;
    try {
      futureCmdResult = cmd.executeAsync();
    } catch (CommandException e) {
      Throwables.throwIfInstanceOf(e.getCause(), IOException.class);
    }

    if (futureCmdResult != null) {
      try {
        cmdResult = futureCmdResult.get();
      } catch (AbnormalTerminationException e) {
        cmdResult = e.getResult();
      }
    }

    long timeoutMillis =
        action.hasTimeout()
            ? Durations.toMillis(action.getTimeout())
            : TimeUnit.MINUTES.toMillis(15);
    boolean wasTimeout =
        (cmdResult != null && cmdResult.getTerminationStatus().timedOut())
            || wasTimeout(timeoutMillis, System.currentTimeMillis() - startTime);
    final int exitCode;
    Status errStatus = null;
    ExecuteResponse.Builder resp = ExecuteResponse.newBuilder();
    if (wasTimeout) {
      final String errMessage =
          String.format(
              "Command:\n%s\nexceeded deadline of %f seconds.",
              Arrays.toString(command.getArgumentsList().toArray()), timeoutMillis / 1000.0);
      logger.warning(errMessage);
      errStatus =
          Status.newBuilder()
              .setCode(Code.DEADLINE_EXCEEDED.getNumber())
              .setMessage(errMessage)
              .build();
      exitCode = LOCAL_EXEC_ERROR;
    } else if (cmdResult == null) {
      exitCode = LOCAL_EXEC_ERROR;
    } else {
      exitCode = cmdResult.getTerminationStatus().getRawExitCode();
    }

    ActionResult.Builder result = ActionResult.newBuilder();
    boolean setResult = exitCode == 0 && !action.getDoNotCache();
    try {
      cache.upload(result, actionKey, action, command, execRoot, outputs, setResult);
    } catch (ExecException e) {
      if (errStatus == null) {
        errStatus =
            Status.newBuilder()
                .setCode(Code.FAILED_PRECONDITION.getNumber())
                .setMessage(e.getMessage())
                .build();
      }
    }
    byte[] stdout = cmdResult.getStdout();
    byte[] stderr = cmdResult.getStderr();
    cache.uploadOutErr(result, stdout, stderr);
    ActionResult finalResult = result.setExitCode(exitCode).build();
    resp.setResult(finalResult);
    if (errStatus != null) {
      resp.setStatus(errStatus);
      throw new ExecutionStatusException(errStatus, resp.build());
    } else if (setResult) {
      cache.setCachedActionResult(actionKey, finalResult);
    }
    return finalResult;
  }

  // Returns true if the OS being run on is Windows (or some close approximation thereof).
  private boolean isWindows() {
    return System.getProperty("os.name").startsWith("Windows");
  }

  private boolean wasTimeout(long timeoutMillis, long wallTimeMillis) {
    return timeoutMillis > 0 && wallTimeMillis > timeoutMillis;
  }

  private Map<String, String> getEnvironmentVariables(Command command) {
    HashMap<String, String> result = new HashMap<>();
    for (EnvironmentVariable v : command.getEnvironmentVariablesList()) {
      result.put(v.getName(), v.getValue());
    }
    return result;
  }

  // Gets the uid of the current user. If uid could not be successfully fetched (e.g., on other
  // platforms, if for some reason the timeout was not met, if "id -u" returned non-numeric
  // number, etc), logs a WARNING and return -1.
  // This is used to set "-u UID" flag for commands running inside Docker containers. There are
  // only a small handful of cases where uid is vital (e.g., if strict permissions are set on the
  // output files), so most use cases would work without setting uid.
  private long getUid() {
    com.google.devtools.build.lib.shell.Command cmd =
        new com.google.devtools.build.lib.shell.Command(
            new String[] {"id", "-u"},
            /*environmentVariables=*/ null,
            /*workingDirectory=*/ null,
            uidTimeout);
    try {
      ByteArrayOutputStream stdout = new ByteArrayOutputStream();
      ByteArrayOutputStream stderr = new ByteArrayOutputStream();
      cmd.execute(stdout, stderr);
      return Long.parseLong(stdout.toString().trim());
    } catch (CommandException | NumberFormatException e) {
      logger.log(
          WARNING, "Could not get UID for passing to Docker container. Proceeding without it.", e);
      return -1;
    }
  }

  // Checks Action for docker container definition. If no docker container specified, returns
  // null. Otherwise returns docker container name from the parameters.
  private String dockerContainer(Command cmd) throws StatusException {
    String result = null;
    for (Platform.Property property : cmd.getPlatform().getPropertiesList()) {
      if (property.getName().equals(CONTAINER_IMAGE_ENTRY_NAME)) {
        if (result != null) {
          // Multiple container name entries
          throw StatusUtils.invalidArgumentError(
              "platform", // Field name.
              String.format(
                  "Multiple entries for %s in action.Platform", CONTAINER_IMAGE_ENTRY_NAME));
        }
        result = property.getValue();
        if (!result.startsWith(DOCKER_IMAGE_PREFIX)) {
          throw StatusUtils.invalidArgumentError(
              "platform", // Field name.
              String.format(
                  "%s: Docker images must be stored in gcr.io with an image spec in the form "
                      + "'docker://gcr.io/{IMAGE_NAME}'",
                  CONTAINER_IMAGE_ENTRY_NAME));
        }
        result = result.substring(DOCKER_IMAGE_PREFIX.length());
      }
    }
    return result;
  }

  // Converts the Command proto into the shell Command object.
  // If no docker container is specified, creates a Command straight from the
  // arguments. Otherwise, returns a Command that would run the specified command inside the
  // specified docker container.
  private com.google.devtools.build.lib.shell.Command getCommand(Command cmd, String pathString)
      throws StatusException {
    Map<String, String> environmentVariables = getEnvironmentVariables(cmd);
    String container = dockerContainer(cmd);
    if (container != null) {
      // Run command inside a docker container.
      ArrayList<String> newCommandLineElements = new ArrayList<>(cmd.getArgumentsCount());
      newCommandLineElements.add("docker");
      newCommandLineElements.add("run");

      // -u doesn't currently make sense for Windows:
      // https://github.com/docker/for-win/issues/636#issuecomment-293653788
      if (!isWindows()) {
        long uid = getUid();
        if (uid >= 0) {
          newCommandLineElements.add("-u");
          newCommandLineElements.add(Long.toString(uid));
        }
      }

      String dockerPathString = pathString + "-docker";
      newCommandLineElements.add("-v");
      newCommandLineElements.add(pathString + ":" + dockerPathString);
      newCommandLineElements.add("-w");
      newCommandLineElements.add(dockerPathString);

      for (Map.Entry<String, String> entry : environmentVariables.entrySet()) {
        String key = entry.getKey();
        String value = entry.getValue();

        newCommandLineElements.add("-e");
        newCommandLineElements.add(key + "=" + value);
      }

      newCommandLineElements.add(container);

      newCommandLineElements.addAll(cmd.getArgumentsList());

      return new com.google.devtools.build.lib.shell.Command(
          newCommandLineElements.toArray(new String[0]), null, new File(pathString));
    } else if (sandboxPath != null) {
      // Run command with sandboxing.
      ArrayList<String> newCommandLineElements = new ArrayList<>(cmd.getArgumentsCount());
      newCommandLineElements.add(sandboxPath.getPathString());
      if (workerOptions.sandboxingBlockNetwork) {
        newCommandLineElements.add("-N");
      }
      for (String writablePath : workerOptions.sandboxingWritablePaths) {
        newCommandLineElements.add("-w");
        newCommandLineElements.add(writablePath);
      }
      for (String tmpfsDir : workerOptions.sandboxingTmpfsDirs) {
        newCommandLineElements.add("-e");
        newCommandLineElements.add(tmpfsDir);
      }
      newCommandLineElements.add("--");
      newCommandLineElements.addAll(cmd.getArgumentsList());
      return new com.google.devtools.build.lib.shell.Command(
          newCommandLineElements.toArray(new String[0]),
          environmentVariables,
          new File(pathString));
    } else {
      // Just run the command.
      return new com.google.devtools.build.lib.shell.Command(
          cmd.getArgumentsList().toArray(new String[0]),
          environmentVariables,
          new File(pathString));
    }
  }
}
