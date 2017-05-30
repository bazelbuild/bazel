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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.remote.CacheNotFoundException;
import com.google.devtools.build.lib.remote.CasServiceGrpc.CasServiceImplBase;
import com.google.devtools.build.lib.remote.ChannelOptions;
import com.google.devtools.build.lib.remote.Chunker;
import com.google.devtools.build.lib.remote.ContentDigests;
import com.google.devtools.build.lib.remote.ContentDigests.ActionKey;
import com.google.devtools.build.lib.remote.ExecuteServiceGrpc.ExecuteServiceImplBase;
import com.google.devtools.build.lib.remote.ExecutionCacheServiceGrpc.ExecutionCacheServiceImplBase;
import com.google.devtools.build.lib.remote.RemoteOptions;
import com.google.devtools.build.lib.remote.RemoteProtocol;
import com.google.devtools.build.lib.remote.RemoteProtocol.Action;
import com.google.devtools.build.lib.remote.RemoteProtocol.ActionResult;
import com.google.devtools.build.lib.remote.RemoteProtocol.BlobChunk;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasStatus;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadTreeMetadataReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadTreeMetadataRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.Command.EnvironmentEntry;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheSetReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheSetRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionCacheStatus;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionStatus;
import com.google.devtools.build.lib.remote.RemoteProtocol.FileNode;
import com.google.devtools.build.lib.remote.RemoteProtocol.Platform;
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
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ProcessUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import com.google.protobuf.Duration;
import com.google.protobuf.util.Durations;
import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
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

/**
 * Implements a remote worker that accepts work items as protobufs. The server implementation is
 * based on grpc.
 */
public class RemoteWorker {
  private static final Logger LOG = Logger.getLogger(RemoteWorker.class.getName());
  private static final boolean LOG_FINER = LOG.isLoggable(Level.FINER);

  private static final int LOCAL_EXEC_ERROR = -1;
  private static final int SIGALRM_EXIT_CODE = /*SIGNAL_BASE=*/128 + /*SIGALRM=*/14;

  private final CasServiceImplBase casServer;
  private final ExecuteServiceImplBase execServer;
  private final ExecutionCacheServiceImplBase execCacheServer;
  private final SimpleBlobStoreActionCache cache;
  private final RemoteWorkerOptions workerOptions;
  private final RemoteOptions remoteOptions;

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
      execServer = new ExecutionServer(workPath);
    } else {
      execServer = null;
    }
    casServer = new CasServer();
    execCacheServer = new ExecutionCacheServer();
  }

  public Server startServer() throws IOException {
    NettyServerBuilder b =
        NettyServerBuilder.forPort(workerOptions.listenPort)
            .maxMessageSize(ChannelOptions.create(Options.getDefaults(AuthAndTLSOptions.class),
                remoteOptions.grpcMaxChunkSizeBytes).maxMessageSize())
            .addService(casServer)
            .addService(execCacheServer);
    if (execServer != null) {
      b.addService(execServer);
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

  class CasServer extends CasServiceImplBase {
    private static final int MAX_MEMORY_KBYTES = 512 * 1024;

    @Override
    public void lookup(CasLookupRequest request, StreamObserver<CasLookupReply> responseObserver) {
      CasLookupReply.Builder reply = CasLookupReply.newBuilder();
      CasStatus.Builder status = reply.getStatusBuilder();
      for (ContentDigest digest : request.getDigestList()) {
        if (!cache.containsKey(digest)) {
          status.addMissingDigest(digest);
        }
      }
      if (status.getMissingDigestCount() > 0) {
        status.setSucceeded(false);
        status.setError(CasStatus.ErrorCode.MISSING_DIGEST);
      } else {
        status.setSucceeded(true);
      }
      responseObserver.onNext(reply.build());
      responseObserver.onCompleted();
    }

    @Override
    public void uploadTreeMetadata(
        CasUploadTreeMetadataRequest request,
        StreamObserver<CasUploadTreeMetadataReply> responseObserver) {
      try {
        for (FileNode treeNode : request.getTreeNodeList()) {
          cache.uploadBlob(treeNode.toByteArray());
        }
        responseObserver.onNext(
            CasUploadTreeMetadataReply.newBuilder()
                .setStatus(CasStatus.newBuilder().setSucceeded(true))
                .build());
      } catch (Exception e) {
        LOG.warning("Request failed: " + e.toString());
        CasUploadTreeMetadataReply.Builder reply = CasUploadTreeMetadataReply.newBuilder();
        reply
            .getStatusBuilder()
            .setSucceeded(false)
            .setError(CasStatus.ErrorCode.UNKNOWN)
            .setErrorDetail(e.toString());
        responseObserver.onNext(reply.build());
      } finally {
        responseObserver.onCompleted();
      }
    }

    @Override
    public void downloadBlob(
        CasDownloadBlobRequest request, StreamObserver<CasDownloadReply> responseObserver) {
      CasDownloadReply.Builder reply = CasDownloadReply.newBuilder();
      CasStatus.Builder status = reply.getStatusBuilder();
      for (ContentDigest digest : request.getDigestList()) {
        if (!cache.containsKey(digest)) {
          status.addMissingDigest(digest);
        }
      }
      if (status.getMissingDigestCount() > 0) {
        status.setSucceeded(false);
        status.setError(CasStatus.ErrorCode.MISSING_DIGEST);
        responseObserver.onNext(reply.build());
        responseObserver.onCompleted();
        return;
      }
      status.setSucceeded(true);
      try {
        // This still relies on the total blob size to be small enough to fit in memory
        // simultaneously! TODO(olaola): refactor to fix this if the need arises.
        Chunker.Builder b = new Chunker.Builder().chunkSize(remoteOptions.grpcMaxChunkSizeBytes);
        for (ContentDigest digest : request.getDigestList()) {
          b.addInput(cache.downloadBlob(digest));
        }
        Chunker c = b.build();
        while (c.hasNext()) {
          reply.setData(c.next());
          responseObserver.onNext(reply.build());
          if (reply.hasStatus()) {
            reply.clearStatus(); // Only send status on first chunk.
          }
        }
      } catch (IOException e) {
        // This cannot happen, as we are chunking in-memory blobs.
        throw new RuntimeException("Internal error: " + e);
      } catch (CacheNotFoundException e) {
        // This can only happen if an item gets evicted right after we check.
        reply.clearData();
        status.setSucceeded(false);
        status.setError(CasStatus.ErrorCode.MISSING_DIGEST);
        status.addMissingDigest(e.getMissingDigest());
        responseObserver.onNext(reply.build());
      } finally {
        responseObserver.onCompleted();
      }
    }

    @Override
    public StreamObserver<CasUploadBlobRequest> uploadBlob(
        final StreamObserver<CasUploadBlobReply> responseObserver) {
      return new StreamObserver<CasUploadBlobRequest>() {
        byte[] blob = null;
        ContentDigest digest = null;
        long offset = 0;

        @Override
        public void onNext(CasUploadBlobRequest request) {
          BlobChunk chunk = request.getData();
          try {
            if (chunk.hasDigest()) {
              // Check if the previous chunk was really done.
              Preconditions.checkArgument(
                  digest == null || offset == 0,
                  "Missing input chunk for digest %s",
                  digest == null ? "" : ContentDigests.toString(digest));
              digest = chunk.getDigest();
              // This unconditionally downloads the whole blob into memory!
              Preconditions.checkArgument((int) (digest.getSizeBytes() / 1024) < MAX_MEMORY_KBYTES);
              blob = new byte[(int) digest.getSizeBytes()];
            }
            Preconditions.checkArgument(digest != null, "First chunk contains no digest");
            Preconditions.checkArgument(
                offset == chunk.getOffset(),
                "Missing input chunk for digest %s",
                ContentDigests.toString(digest));
            if (digest.getSizeBytes() > 0) {
              chunk.getData().copyTo(blob, (int) offset);
              offset = (offset + chunk.getData().size()) % digest.getSizeBytes();
            }
            if (offset == 0) {
              ContentDigest uploadedDigest = cache.uploadBlob(blob);
              Preconditions.checkArgument(
                  uploadedDigest.equals(digest),
                  "Digest mismatch: client sent %s, server computed %s",
                  ContentDigests.toString(digest),
                  ContentDigests.toString(uploadedDigest));
            }
          } catch (Exception e) {
            LOG.warning("Request failed: " + e.toString());
            CasUploadBlobReply.Builder reply = CasUploadBlobReply.newBuilder();
            reply
                .getStatusBuilder()
                .setSucceeded(false)
                .setError(
                    e instanceof IllegalArgumentException
                        ? CasStatus.ErrorCode.INVALID_ARGUMENT
                        : CasStatus.ErrorCode.UNKNOWN)
                .setErrorDetail(e.toString());
            responseObserver.onNext(reply.build());
          }
        }

        @Override
        public void onError(Throwable t) {
          LOG.warning("Request errored remotely: " + t);
        }

        @Override
        public void onCompleted() {
          responseObserver.onCompleted();
        }
      };
    }
  }

  class ExecutionCacheServer extends ExecutionCacheServiceImplBase {
    @Override
    public void getCachedResult(
        ExecutionCacheRequest request, StreamObserver<ExecutionCacheReply> responseObserver) {
      try {
        ActionKey actionKey = ContentDigests.unsafeActionKeyFromDigest(request.getActionDigest());
        ExecutionCacheReply.Builder reply = ExecutionCacheReply.newBuilder();
        ActionResult result = cache.getCachedActionResult(actionKey);
        if (result != null) {
          reply.setResult(result);
        }
        reply.getStatusBuilder().setSucceeded(true);
        responseObserver.onNext(reply.build());
      } catch (Exception e) {
        LOG.warning("getCachedActionResult request failed: " + e.toString());
        ExecutionCacheReply.Builder reply = ExecutionCacheReply.newBuilder();
        reply
            .getStatusBuilder()
            .setSucceeded(false)
            .setError(ExecutionCacheStatus.ErrorCode.UNKNOWN);
        responseObserver.onNext(reply.build());
      } finally {
        responseObserver.onCompleted();
      }
    }

    @Override
    public void setCachedResult(
        ExecutionCacheSetRequest request, StreamObserver<ExecutionCacheSetReply> responseObserver) {
      try {
        ActionKey actionKey = ContentDigests.unsafeActionKeyFromDigest(request.getActionDigest());
        cache.setCachedActionResult(actionKey, request.getResult());
        ExecutionCacheSetReply.Builder reply = ExecutionCacheSetReply.newBuilder();
        reply.getStatusBuilder().setSucceeded(true);
        responseObserver.onNext(reply.build());
      } catch (Exception e) {
        LOG.warning("setCachedActionResult request failed: " + e.toString());
        ExecutionCacheSetReply.Builder reply = ExecutionCacheSetReply.newBuilder();
        reply
            .getStatusBuilder()
            .setSucceeded(false)
            .setError(ExecutionCacheStatus.ErrorCode.UNKNOWN);
        responseObserver.onNext(reply.build());
      } finally {
        responseObserver.onCompleted();
      }
    }
  }

  // How long to wait for the uid command.
  private static final Duration uidTimeout = Durations.fromMicros(30);

  class ExecutionServer extends ExecuteServiceImplBase {
    private final Path workPath;

    //The name of the container image entry in the Platform proto
    // (see src/main/protobuf/remote_protocol.proto and
    // experimental_remote_platform_override in
    // src/main/java/com/google/devtools/build/lib/remote/RemoteOptions.java)
    public static final String CONTAINER_IMAGE_ENTRY_NAME = "container-image";

    public ExecutionServer(Path workPath) {
      this.workPath = workPath;
    }

    private Map<String, String> getEnvironmentVariables(RemoteProtocol.Command command) {
      HashMap<String, String> result = new HashMap<>();
      for (EnvironmentEntry entry : command.getEnvironmentList()) {
        result.put(entry.getVariable(), entry.getValue());
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
      List<Platform.Property> entries = action.getPlatform().getEntryList();

      for (Platform.Property entry : entries) {
        if (entry.getName().equals(CONTAINER_IMAGE_ENTRY_NAME)) {
          if (result == null) {
            result = entry.getValue();
          } else {
            // Multiple container name entries
            throw new IllegalArgumentException(
                "Multiple entries for " + CONTAINER_IMAGE_ENTRY_NAME + " in action.Platform");
          }
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

    public ExecuteReply execute(Action action, Path execRoot)
        throws IOException, InterruptedException, IllegalArgumentException {
      ByteArrayOutputStream stdout = new ByteArrayOutputStream();
      ByteArrayOutputStream stderr = new ByteArrayOutputStream();
      RemoteProtocol.Command command;
      try {
        command =
            RemoteProtocol.Command.parseFrom(cache.downloadBlob(action.getCommandDigest()));
        cache.downloadTree(action.getInputRootDigest(), execRoot);
      } catch (CacheNotFoundException e) {
        LOG.warning("Cache miss on " + ContentDigests.toString(e.getMissingDigest()));
        return ExecuteReply.newBuilder()
            .setCasError(
                CasStatus.newBuilder()
                    .setSucceeded(false)
                    .addMissingDigest(e.getMissingDigest())
                    .setError(CasStatus.ErrorCode.MISSING_DIGEST)
                    .setErrorDetail(e.toString()))
            .setStatus(
                ExecutionStatus.newBuilder()
                    .setExecuted(false)
                    .setSucceeded(false)
                    .setError(
                        e.getMissingDigest() == action.getCommandDigest()
                            ? ExecutionStatus.ErrorCode.MISSING_COMMAND
                            : ExecutionStatus.ErrorCode.MISSING_INPUT)
                    .setErrorDetail(e.toString()))
            .build();
      }

      List<Path> outputs = new ArrayList<>(action.getOutputPathList().size());
      for (String output : action.getOutputPathList()) {
        Path file = execRoot.getRelative(output);
        if (file.exists()) {
          throw new FileAlreadyExistsException("Output file already exists: " + file);
        }
        FileSystemUtils.createDirectoryAndParents(file.getParentDirectory());
        outputs.add(file);
      }

      // TODO(ulfjack): This is basically a copy of LocalSpawnRunner. Ideally, we'd use that
      // implementation instead of copying it.
      // TODO(ulfjack): Timeout is specified in ExecuteRequest, but not passed in yet.
      int timeoutSeconds = 60 * 15;
      Command cmd =
          getCommand(
              action,
              command.getArgvList().toArray(new String[] {}),
              getEnvironmentVariables(command),
              execRoot.getPathString());

      long startTime = System.currentTimeMillis();
      CommandResult cmdResult;
      try {
        cmdResult = cmd.execute(Command.NO_INPUT, Command.NO_OBSERVER, stdout, stderr, true);
      } catch (AbnormalTerminationException e) {
        cmdResult = e.getResult();
      } catch (CommandException e) {
        // At the time this comment was written, this must be a ExecFailedException encapsulating an
        // IOException from the underlying Subprocess.Factory.
        LOG.warning("Execution failed for " + command.getArgvList());
        return ExecuteReply.newBuilder()
            .setResult(
                ActionResult.newBuilder()
                    .setReturnCode(LOCAL_EXEC_ERROR))
            .setStatus(
                ExecutionStatus.newBuilder()
                    .setExecuted(false)
                    .setSucceeded(false)
                    .setError(ExecutionStatus.ErrorCode.EXEC_FAILED)
                    .setErrorDetail(e.toString()))
            .build();
      }
      long wallTime = System.currentTimeMillis() - startTime;
      boolean wasTimeout = cmdResult.getTerminationStatus().timedout()
          || wasTimeout(timeoutSeconds, wallTime);
      Status status = wasTimeout ? Status.TIMEOUT : Status.SUCCESS;
      int exitCode = status == Status.TIMEOUT
          ? SIGALRM_EXIT_CODE
          : cmdResult.getTerminationStatus().getRawExitCode();

      ImmutableList<ContentDigest> outErrDigests =
          cache.uploadBlobs(ImmutableList.of(stdout.toByteArray(), stderr.toByteArray()));
      ContentDigest stdoutDigest = outErrDigests.get(0);
      ContentDigest stderrDigest = outErrDigests.get(1);
      ActionResult.Builder actionResult =
          ActionResult.newBuilder()
              .setReturnCode(exitCode)
              .setStdoutDigest(stdoutDigest)
              .setStderrDigest(stderrDigest);
      cache.uploadAllResults(execRoot, outputs, actionResult);
      cache.setCachedActionResult(ContentDigests.computeActionKey(action), actionResult.build());
      return ExecuteReply.newBuilder()
          .setResult(actionResult)
          .setStatus(ExecutionStatus.newBuilder().setExecuted(true).setSucceeded(true))
          .build();
    }

    private boolean wasTimeout(int timeoutSeconds, long wallTimeMillis) {
      return timeoutSeconds > 0 && wallTimeMillis / 1000.0 > timeoutSeconds;
    }

    @Override
    public void execute(ExecuteRequest request, StreamObserver<ExecuteReply> responseObserver) {
      Path tempRoot = workPath.getRelative("build-" + UUID.randomUUID().toString());
      try {
        tempRoot.createDirectory();
        if (LOG_FINER) {
          LOG.fine(
              "Work received has "
                  + request.getTotalInputFileCount()
                  + " input files and "
                  + request.getAction().getOutputPathCount()
                  + " output files.");
        }
        ExecuteReply reply = execute(request.getAction(), tempRoot);
        responseObserver.onNext(reply);
        if (workerOptions.debug) {
          if (!reply.getStatus().getSucceeded()) {
            LOG.warning("Work failed. Request: " + request.toString() + ".");
          } else if (LOG_FINER) {
            LOG.fine("Work completed.");
          }
        }
        if (!workerOptions.debug) {
          FileSystemUtils.deleteTree(tempRoot);
        } else {
          LOG.warning("Preserving work directory " + tempRoot.toString() + ".");
        }
      } catch (IOException | InterruptedException e) {
        LOG.log(Level.SEVERE, "Failure", e);
        ExecuteReply.Builder reply = ExecuteReply.newBuilder();
        reply.getStatusBuilder().setSucceeded(false).setErrorDetail(e.toString());
        responseObserver.onNext(reply.build());
        if (e instanceof InterruptedException) {
          Thread.currentThread().interrupt();
        }
      } finally {
        responseObserver.onCompleted();
      }
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
