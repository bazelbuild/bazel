// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.remote;

import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.devtools.build.lib.remote.CacheNotFoundException;
import com.google.devtools.build.lib.remote.Chunker;
import com.google.devtools.build.lib.remote.Digests;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.remote.RemoteOptions;
import com.google.devtools.build.lib.remote.SimpleBlobStore;
import com.google.devtools.build.lib.remote.SimpleBlobStoreActionCache;
import com.google.devtools.build.lib.remote.SimpleBlobStoreFactory;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.shell.TimeoutKillableObserver;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.ProcessUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.remoteexecution.v1test.Action;
import com.google.devtools.remoteexecution.v1test.ActionCacheGrpc.ActionCacheImplBase;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.BatchUpdateBlobsRequest;
import com.google.devtools.remoteexecution.v1test.BatchUpdateBlobsResponse;
import com.google.devtools.remoteexecution.v1test.Command.EnvironmentVariable;
import com.google.devtools.remoteexecution.v1test.ContentAddressableStorageGrpc.ContentAddressableStorageImplBase;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.ExecuteResponse;
import com.google.devtools.remoteexecution.v1test.ExecutionGrpc.ExecutionImplBase;
import com.google.devtools.remoteexecution.v1test.FindMissingBlobsRequest;
import com.google.devtools.remoteexecution.v1test.FindMissingBlobsResponse;
import com.google.devtools.remoteexecution.v1test.GetActionResultRequest;
import com.google.devtools.remoteexecution.v1test.Platform;
import com.google.devtools.remoteexecution.v1test.UpdateActionResultRequest;
import com.google.devtools.remoteexecution.v1test.UpdateBlobRequest;
import com.google.longrunning.Operation;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.Duration;
import com.google.protobuf.util.Durations;
import com.google.rpc.BadRequest;
import com.google.rpc.BadRequest.FieldViolation;
import com.google.rpc.Code;
import com.google.rpc.Status;
import com.google.watcher.v1.Change;
import com.google.watcher.v1.ChangeBatch;
import com.google.watcher.v1.Request;
import com.google.watcher.v1.WatcherGrpc.WatcherImplBase;
import io.grpc.Server;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.protobuf.StatusProto;
import io.grpc.stub.StreamObserver;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.FileAlreadyExistsException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * Implements a remote worker that accepts work items as protobufs. The server implementation is
 * based on grpc.
 */
public class RemoteWorker {
  private static final Logger LOG = Logger.getLogger(RemoteWorker.class.getName());
  private static final boolean LOG_FINER = LOG.isLoggable(Level.FINER);
  private final ContentAddressableStorageImplBase casServer;
  private final ByteStreamImplBase bsServer;
  private final WatcherImplBase watchServer;
  private final ExecutionImplBase execServer;
  private final ActionCacheImplBase actionCacheServer;
  private final SimpleBlobStoreActionCache cache;
  private final RemoteWorkerOptions workerOptions;
  private final RemoteOptions remoteOptions;
  private final ConcurrentHashMap<String, ExecuteRequest> operationsCache;

  public RemoteWorker(
      RemoteWorkerOptions workerOptions,
      RemoteOptions remoteOptions,
      SimpleBlobStoreActionCache cache)
      throws IOException {
    this.cache = cache;
    this.workerOptions = workerOptions;
    this.remoteOptions = remoteOptions;
    if (workerOptions.workPath != null) {
      Path workPath = getFileSystem().getPath(workerOptions.workPath);
      FileSystemUtils.createDirectoryAndParents(workPath);
      watchServer = new WatcherServer(workPath);
      execServer = new ExecutionServer();
    } else {
      watchServer = null;
      execServer = null;
    }
    casServer = new CasServer();
    bsServer = new ByteStreamServer();
    actionCacheServer = new ActionCacheServer();
    operationsCache = new ConcurrentHashMap<>();
  }

  public Server startServer() throws IOException {
    NettyServerBuilder b =
        NettyServerBuilder.forPort(workerOptions.listenPort)
            .addService(casServer)
            .addService(bsServer)
            .addService(actionCacheServer);
    if (execServer != null) {
      b.addService(execServer);
      b.addService(watchServer);
    } else {
      System.out.println("*** Execution disabled, only serving cache requests.");
    }
    Server server = b.build();
    System.out.println(
        "*** Starting grpc server on all locally bound IPs on port "
            + workerOptions.listenPort
            + ".");
    server.start();
    return server;
  }

  private static @Nullable Digest parseDigestFromResourceName(String resourceName) {
    try {
      String[] tokens = resourceName.split("/");
      if (tokens.length < 2) {
        return null;
      }
      String hash = tokens[tokens.length - 2];
      long size = Long.parseLong(tokens[tokens.length - 1]);
      return Digests.buildDigest(hash, size);
    } catch (NumberFormatException e) {
      return null;
    }
  }

  private static StatusRuntimeException internalError(Exception e) {
    return StatusProto.toStatusRuntimeException(internalErrorStatus(e));
  }

  private static com.google.rpc.Status internalErrorStatus(Exception e) {
    return Status.newBuilder()
        .setCode(Code.INTERNAL.getNumber())
        .setMessage("Internal error: " + e)
        .build();
  }

  private static StatusRuntimeException notFoundError(Digest digest) {
    return StatusProto.toStatusRuntimeException(notFoundStatus(digest));
  }

  private static com.google.rpc.Status notFoundStatus(Digest digest) {
    return Status.newBuilder()
        .setCode(Code.NOT_FOUND.getNumber())
        .setMessage("Digest not found:" + digest)
        .build();
  }

  private static StatusRuntimeException invalidArgumentError(String field, String desc) {
    return StatusProto.toStatusRuntimeException(invalidArgumentStatus(field, desc));
  }

  private static com.google.rpc.Status invalidArgumentStatus(String field, String desc) {
    FieldViolation v = FieldViolation.newBuilder().setField(field).setDescription(desc).build();
    return Status.newBuilder()
        .setCode(Code.INVALID_ARGUMENT.getNumber())
        .setMessage("invalid argument(s): " + field + ": " + desc)
        .addDetails(Any.pack(BadRequest.newBuilder().addFieldViolations(v).build()))
        .build();
  }

  class CasServer extends ContentAddressableStorageImplBase {
    @Override
    public void findMissingBlobs(
        FindMissingBlobsRequest request,
        StreamObserver<FindMissingBlobsResponse> responseObserver) {
      FindMissingBlobsResponse.Builder response = FindMissingBlobsResponse.newBuilder();
      for (Digest digest : request.getBlobDigestsList()) {
        if (!cache.containsKey(digest)) {
          response.addMissingBlobDigests(digest);
        }
      }
      responseObserver.onNext(response.build());
      responseObserver.onCompleted();
    }

    @Override
    public void batchUpdateBlobs(
        BatchUpdateBlobsRequest request,
        StreamObserver<BatchUpdateBlobsResponse> responseObserver) {
      BatchUpdateBlobsResponse.Builder batchResponse = BatchUpdateBlobsResponse.newBuilder();
      for (UpdateBlobRequest r : request.getRequestsList()) {
        BatchUpdateBlobsResponse.Response.Builder resp = batchResponse.addResponsesBuilder();
        try {
          Digest digest = cache.uploadBlob(r.getData().toByteArray());
          if (!r.getContentDigest().equals(digest)) {
            String err =
                "Upload digest " + r.getContentDigest() + " did not match data digest: " + digest;
            resp.setStatus(invalidArgumentStatus("content_digest", err));
            continue;
          }
          resp.getStatusBuilder().setCode(Code.OK.getNumber());
        } catch (Exception e) {
          resp.setStatus(internalErrorStatus(e));
        }
      }
      responseObserver.onNext(batchResponse.build());
      responseObserver.onCompleted();
    }
  }

  class ByteStreamServer extends ByteStreamImplBase {
    @Override
    public void read(ReadRequest request, StreamObserver<ReadResponse> responseObserver) {
      Digest digest = parseDigestFromResourceName(request.getResourceName());
      if (digest == null) {
        responseObserver.onError(
            invalidArgumentError(
                "resource_name",
                "Failed parsing digest from resource_name:" + request.getResourceName()));
      }
      if (!cache.containsKey(digest)) {
        responseObserver.onError(notFoundError(digest));
        return;
      }
      try {
        // This still relies on the blob size to be small enough to fit in memory.
        // TODO(olaola): refactor to fix this if the need arises.
        Chunker c = Chunker.from(cache.downloadBlob(digest));
        while (c.hasNext()) {
          responseObserver.onNext(
              ReadResponse.newBuilder().setData(ByteString.copyFrom(c.next().getData())).build());
        }
        responseObserver.onCompleted();
      } catch (CacheNotFoundException e) {
        // This can only happen if an item gets evicted right after we check.
        responseObserver.onError(notFoundError(digest));
      } catch (Exception e) {
        LOG.warning("Read request failed: " + e);
        responseObserver.onError(internalError(e));
      }
    }

    @Override
    public StreamObserver<WriteRequest> write(
        final StreamObserver<WriteResponse> responseObserver) {
      return new StreamObserver<WriteRequest>() {
        byte[] blob = null;
        Digest digest = null;
        long offset = 0;
        String resourceName = null;
        boolean closed = false;

        @Override
        public void onNext(WriteRequest request) {
          if (closed) {
            return;
          }
          if (digest == null) {
            resourceName = request.getResourceName();
            digest = parseDigestFromResourceName(resourceName);
            blob = new byte[(int) digest.getSizeBytes()];
          }
          if (digest == null) {
            responseObserver.onError(
                invalidArgumentError(
                    "resource_name",
                    "Failed parsing digest from resource_name:" + request.getResourceName()));
            closed = true;
            return;
          }
          if (request.getWriteOffset() != offset) {
            responseObserver.onError(
                invalidArgumentError(
                    "write_offset",
                    "Expected:" + offset + ", received: " + request.getWriteOffset()));
            closed = true;
            return;
          }
          if (!request.getResourceName().isEmpty()
              && !request.getResourceName().equals(resourceName)) {
            responseObserver.onError(
                invalidArgumentError(
                    "resource_name",
                    "Expected:" + resourceName + ", received: " + request.getResourceName()));
            closed = true;
            return;
          }
          long size = request.getData().size();
          if (size > 0) {
            request.getData().copyTo(blob, (int) offset);
            offset += size;
          }
          boolean shouldFinishWrite = offset == digest.getSizeBytes();
          if (shouldFinishWrite != request.getFinishWrite()) {
            responseObserver.onError(
                invalidArgumentError(
                    "finish_write",
                    "Expected:" + shouldFinishWrite + ", received: " + request.getFinishWrite()));
            closed = true;
          }
        }

        @Override
        public void onError(Throwable t) {
          LOG.warning("Write request errored remotely: " + t);
          closed = true;
        }

        @Override
        public void onCompleted() {
          if (closed) {
            return;
          }
          if (digest == null || offset != digest.getSizeBytes()) {
            responseObserver.onError(
                StatusProto.toStatusRuntimeException(
                    Status.newBuilder()
                        .setCode(Code.FAILED_PRECONDITION.getNumber())
                        .setMessage("Request completed before all data was sent.")
                        .build()));
            closed = true;
            return;
          }
          try {
            Digest d = cache.uploadBlob(blob);
            if (!d.equals(digest)) {
              String err = "Received digest " + digest + " does not match computed digest " + d;
              responseObserver.onError(invalidArgumentError("resource_name", err));
              closed = true;
              return;
            }
            responseObserver.onNext(WriteResponse.newBuilder().setCommittedSize(offset).build());
            responseObserver.onCompleted();
          } catch (Exception e) {
            LOG.warning("Write request failed: " + e);
            responseObserver.onError(internalError(e));
            closed = true;
          }
        }
      };
    }
  }

  class ActionCacheServer extends ActionCacheImplBase {
    @Override
    public void getActionResult(
        GetActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
      try {
        ActionKey actionKey = Digests.unsafeActionKeyFromDigest(request.getActionDigest());
        ActionResult result = cache.getCachedActionResult(actionKey);
        if (result == null) {
          responseObserver.onError(notFoundError(request.getActionDigest()));
          return;
        }
        responseObserver.onNext(result);
        responseObserver.onCompleted();
      } catch (Exception e) {
        LOG.warning("getActionResult request failed: " + e);
        responseObserver.onError(internalError(e));
      }
    }

    @Override
    public void updateActionResult(
        UpdateActionResultRequest request, StreamObserver<ActionResult> responseObserver) {
      try {
        ActionKey actionKey = Digests.unsafeActionKeyFromDigest(request.getActionDigest());
        cache.setCachedActionResult(actionKey, request.getActionResult());
        responseObserver.onNext(request.getActionResult());
        responseObserver.onCompleted();
      } catch (Exception e) {
        LOG.warning("updateActionResult request failed: " + e);
        responseObserver.onError(internalError(e));
      }
    }
  }

  // How long to wait for the uid command.
  private static final Duration uidTimeout = Durations.fromMicros(30);

  class WatcherServer extends WatcherImplBase {
    private final Path workPath;

    //The name of the container image entry in the Platform proto
    // (see third_party/googleapis/devtools/remoteexecution/*/remote_execution.proto and
    // experimental_remote_platform_override in
    // src/main/java/com/google/devtools/build/lib/remote/RemoteOptions.java)
    public static final String CONTAINER_IMAGE_ENTRY_NAME = "container-image";

    private static final int LOCAL_EXEC_ERROR = -1;

    public WatcherServer(Path workPath) {
      this.workPath = workPath;
    }

    private Map<String, String> getEnvironmentVariables(
        com.google.devtools.remoteexecution.v1test.Command command) {
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
      Command cmd = new Command(new String[] {"id", "-u"});
      try {
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        ByteArrayOutputStream stderr = new ByteArrayOutputStream();
        cmd.execute(
            Command.NO_INPUT,
            new TimeoutKillableObserver(Durations.toMicros(uidTimeout)),
            stdout,
            stderr);
        return Long.parseLong(stdout.toString().trim());
      } catch (CommandException | NumberFormatException e) {
        LOG.warning("Could not get UID for passing to Docker container. Proceeding without it.");
        LOG.warning("Error: " + e.toString());
        return -1;
      }
    }

    // Checks Action for docker container definition. If no docker container specified, returns
    // null. Otherwise returns docker container name from the parameters.
    private String dockerContainer(Action action) throws IllegalArgumentException {
      String result = null;
      for (Platform.Property property : action.getPlatform().getPropertiesList()) {
        if (property.getName().equals(CONTAINER_IMAGE_ENTRY_NAME)) {
          if (result != null) {
            // Multiple container name entries
            throw new IllegalArgumentException(
                "Multiple entries for " + CONTAINER_IMAGE_ENTRY_NAME + " in action.Platform");
          }
          result = property.getValue();
        }
      }
      return result;
    }

    // Takes an Action and parameters that can be used to create a Command. Returns the Command.
    // If no docker container is specified inside Action, creates a Command straight from the
    // arguments. Otherwise, returns a Command that would run the specified command inside the
    // specified docker container.
    private Command getCommand(
        Action action,
        String[] commandLineElements,
        Map<String, String> environmentVariables,
        String pathString)
        throws IllegalArgumentException {
      String container = dockerContainer(action);
      if (container == null) {
        // Was not asked to Dokerize.
        return new Command(commandLineElements, environmentVariables, new File(pathString));
      }

      // Run command inside a docker container.
      ArrayList<String> newCommandLineElements = new ArrayList<String>();
      newCommandLineElements.add("docker");
      newCommandLineElements.add("run");

      long uid = getUid();
      if (uid >= 0) {
        newCommandLineElements.add("-u");
        newCommandLineElements.add(Long.toString(uid));
      }

      final String dockerPathString = pathString + "-docker";
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

      newCommandLineElements.addAll(Arrays.asList(commandLineElements));

      return new Command(
          newCommandLineElements.toArray(new String[newCommandLineElements.size()]),
          null,
          new File(pathString));
    }

    static final int MAX_BLOB_SIZE_FOR_INLINE = 1024 * 10;

    private void passOutErr(byte[] stdout, byte[] stderr, ActionResult.Builder result)
        throws InterruptedException {
      if (stdout.length <= MAX_BLOB_SIZE_FOR_INLINE) {
        result.setStdoutRaw(ByteString.copyFrom(stdout));
      } else if (stdout.length > 0) {
        result.setStdoutDigest(cache.uploadBlob(stdout));
      }
      if (stderr.length <= MAX_BLOB_SIZE_FOR_INLINE) {
        result.setStderrRaw(ByteString.copyFrom(stderr));
      } else if (stderr.length > 0) {
        result.setStderrDigest(cache.uploadBlob(stderr));
      }
    }

    public ActionResult execute(Action action, Path execRoot)
        throws IOException, InterruptedException, IllegalArgumentException, CacheNotFoundException {
      ByteArrayOutputStream stdout = new ByteArrayOutputStream();
      ByteArrayOutputStream stderr = new ByteArrayOutputStream();
      ActionResult.Builder result = ActionResult.newBuilder();
      com.google.devtools.remoteexecution.v1test.Command command =
          com.google.devtools.remoteexecution.v1test.Command.parseFrom(
              cache.downloadBlob(action.getCommandDigest()));
      cache.downloadTree(action.getInputRootDigest(), execRoot);

      List<Path> outputs = new ArrayList<>(action.getOutputFilesList().size());
      for (String output : action.getOutputFilesList()) {
        Path file = execRoot.getRelative(output);
        if (file.exists()) {
          throw new FileAlreadyExistsException("Output file already exists: " + file);
        }
        FileSystemUtils.createDirectoryAndParents(file.getParentDirectory());
        outputs.add(file);
      }
      // TODO(olaola): support output directories.

      // TODO(ulfjack): This is basically a copy of LocalSpawnRunner. Ideally, we'd use that
      // implementation instead of copying it.
      Command cmd =
          getCommand(
              action,
              command.getArgumentsList().toArray(new String[] {}),
              getEnvironmentVariables(command),
              execRoot.getPathString());
      long startTime = System.currentTimeMillis();
      CommandResult cmdResult = null;
      try {
        cmdResult = cmd.execute(Command.NO_INPUT, Command.NO_OBSERVER, stdout, stderr, true);
      } catch (AbnormalTerminationException e) {
        cmdResult = e.getResult();
      } catch (CommandException e) {
        // At the time this comment was written, this must be a ExecFailedException encapsulating
        // an IOException from the underlying Subprocess.Factory.
      }
      final int timeoutSeconds = 60 * 15;
      // TODO(ulfjack): Timeout is specified in ExecuteRequest, but not passed in yet.
      boolean wasTimeout =
          cmdResult != null && cmdResult.getTerminationStatus().timedout()
              || wasTimeout(timeoutSeconds, System.currentTimeMillis() - startTime);
      int exitCode;
      if (wasTimeout) {
        final String errMessage =
            "Command:\n"
                + command.getArgumentsList()
                + "\nexceeded deadline of "
                + timeoutSeconds
                + "seconds";
        LOG.warning(errMessage);
        throw StatusProto.toStatusRuntimeException(
            Status.newBuilder()
                .setCode(Code.DEADLINE_EXCEEDED.getNumber())
                .setMessage(errMessage)
                .build());
      } else if (cmdResult == null) {
        exitCode = LOCAL_EXEC_ERROR;
      } else {
        exitCode = cmdResult.getTerminationStatus().getRawExitCode();
      }

      passOutErr(stdout.toByteArray(), stderr.toByteArray(), result);
      cache.uploadAllResults(execRoot, outputs, result);
      ActionResult finalResult = result.setExitCode(exitCode).build();
      if (exitCode == 0) {
        cache.setCachedActionResult(Digests.computeActionKey(action), finalResult);
      }
      return finalResult;
    }

    private boolean wasTimeout(int timeoutSeconds, long wallTimeMillis) {
      return timeoutSeconds > 0 && wallTimeMillis / 1000.0 > timeoutSeconds;
    }

    @Override
    public void watch(Request wr, StreamObserver<ChangeBatch> responseObserver) {
      final String opName = wr.getTarget();
      if (!operationsCache.containsKey(opName)) {
        responseObserver.onError(
            StatusProto.toStatusRuntimeException(
                Status.newBuilder()
                    .setCode(Code.NOT_FOUND.getNumber())
                    .setMessage("Operation not found: " + opName)
                    .build()));
      }
      ExecuteRequest request = operationsCache.get(opName);
      Path tempRoot = workPath.getRelative("build-" + opName);
      try {
        tempRoot.createDirectory();
        if (LOG_FINER) {
          LOG.fine(
              "Work received has "
                  + request.getTotalInputFileCount()
                  + " input files and "
                  + request.getAction().getOutputFilesCount()
                  + " output files.");
        }
        ActionResult result = execute(request.getAction(), tempRoot);
        responseObserver.onNext(
            ChangeBatch.newBuilder()
                .addChanges(
                    Change.newBuilder()
                        .setState(Change.State.EXISTS)
                        .setData(
                            Any.pack(
                                Operation.newBuilder()
                                    .setName(opName)
                                    .setDone(true)
                                    .setResponse(
                                        Any.pack(
                                            ExecuteResponse.newBuilder()
                                                .setResult(result)
                                                .build()))
                                    .build()))
                        .build())
                .build());
        responseObserver.onCompleted();
      } catch (CacheNotFoundException e) {
        LOG.warning("Cache miss on " + e.getMissingDigest());
        responseObserver.onError(notFoundError(e.getMissingDigest()));
      } catch (StatusRuntimeException e) {
        responseObserver.onError(e);
      } catch (IllegalArgumentException e) {
        responseObserver.onError(
            StatusProto.toStatusRuntimeException(
                Status.newBuilder()
                    .setCode(Code.INVALID_ARGUMENT.getNumber())
                    .setMessage(e.toString())
                    .build()));
      } catch (Exception e) {
        StringWriter stringWriter = new StringWriter();
        e.printStackTrace(new PrintWriter(stringWriter));
        LOG.log(Level.SEVERE, "Work failed: " + e + stringWriter.toString());
        responseObserver.onError(internalError(e));
        if (e instanceof InterruptedException) {
          Thread.currentThread().interrupt();
        }
      } finally {
        operationsCache.remove(opName);
        if (workerOptions.debug) {
          LOG.warning("Preserving work directory " + tempRoot);
        } else {
          try {
            FileSystemUtils.deleteTree(tempRoot);
          } catch (IOException e) {
            throw new RuntimeException(
                String.format("Failed to delete tmp directory %s: %s: ", tempRoot, e));
          }
        }
      }
    }
  }

  class ExecutionServer extends ExecutionImplBase {
    @Override
    public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
      // Defer the actual action execution to the Watcher.watch request.
      // There are a lot of errors for which we could fail early here, but deferring them all
      // is simpler.
      final String opName = UUID.randomUUID().toString();
      operationsCache.put(opName, request);
      responseObserver.onNext(Operation.newBuilder().setName(opName).build());
      responseObserver.onCompleted();
    }
  }

  public static void main(String[] args) throws Exception {
    OptionsParser parser =
        OptionsParser.newOptionsParser(RemoteOptions.class, RemoteWorkerOptions.class);
    parser.parseAndExitUponError(args);
    RemoteOptions remoteOptions = parser.getOptions(RemoteOptions.class);
    RemoteWorkerOptions remoteWorkerOptions = parser.getOptions(RemoteWorkerOptions.class);

    System.out.println("*** Initializing in-memory cache server.");
    boolean remoteCache = SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions);
    if (!remoteCache) {
      System.out.println("*** Not using remote cache. This should be used for testing only!");
    }
    SimpleBlobStore blobStore =
        remoteCache
            ? SimpleBlobStoreFactory.create(remoteOptions)
            : new SimpleBlobStoreFactory.ConcurrentMapBlobStore(
                new ConcurrentHashMap<String, byte[]>());

    RemoteWorker worker =
        new RemoteWorker(
            remoteWorkerOptions, remoteOptions, new SimpleBlobStoreActionCache(blobStore));
    final Server server = worker.startServer();

    final Path pidFile;
    if (remoteWorkerOptions.pidFile != null) {
      pidFile = getFileSystem().getPath(remoteWorkerOptions.pidFile);
      PrintWriter writer = new PrintWriter(pidFile.getOutputStream());
      writer.append(Integer.toString(ProcessUtils.getpid()));
      writer.append("\n");
      writer.close();
    } else {
      pidFile = null;
    }

    Runtime.getRuntime()
        .addShutdownHook(
            new Thread() {
              @Override
              public void run() {
                System.err.println("*** Shutting down grpc server.");
                server.shutdown();
                if (pidFile != null) {
                  try {
                    pidFile.delete();
                  } catch (IOException e) {
                    System.err.println("Cannot remove pid file: " + pidFile.toString());
                  }
                }
                System.err.println("*** Server shut down.");
              }
            });
    server.awaitTermination();
  }

  static FileSystem getFileSystem() {
    return OS.getCurrent() == OS.WINDOWS ? new JavaIoFileSystem() : new UnixFileSystem();
  }
}
