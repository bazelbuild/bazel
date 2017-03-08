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
import com.google.devtools.build.lib.remote.CacheNotFoundException;
import com.google.devtools.build.lib.remote.CasServiceGrpc.CasServiceImplBase;
import com.google.devtools.build.lib.remote.ConcurrentMapActionCache;
import com.google.devtools.build.lib.remote.ConcurrentMapFactory;
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
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ProcessUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParser;
import com.google.protobuf.ByteString;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.FileAlreadyExistsException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Implements a remote worker that accepts work items as protobufs. The server implementation is
 * based on grpc.
 */
public class RemoteWorker {
  private static final Logger LOG = Logger.getLogger(RemoteWorker.class.getName());
  private static final boolean LOG_FINER = LOG.isLoggable(Level.FINER);
  private final ConcurrentMapActionCache cache;
  private final CasServiceImplBase casServer;
  private final ExecuteServiceImplBase execServer;
  private final ExecutionCacheServiceImplBase execCacheServer;

  public RemoteWorker(Path workPath, RemoteWorkerOptions options, ConcurrentMapActionCache cache) {
    this.cache = cache;
    casServer = new CasServer();
    execServer = new ExecutionServer(workPath, options);
    execCacheServer = new ExecutionCacheServer();
  }

  public CasServiceImplBase getCasServer() {
    return casServer;
  }

  public ExecuteServiceImplBase getExecutionServer() {
    return execServer;
  }

  public ExecutionCacheServiceImplBase getExecCacheServer() {
    return execCacheServer;
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
        for (ContentDigest digest : request.getDigestList()) {
          reply.setData(
              BlobChunk.newBuilder()
                  .setDigest(digest)
                  .setData(ByteString.copyFrom(cache.downloadBlob(digest)))
                  .build());
          responseObserver.onNext(reply.build());
          if (reply.hasStatus()) {
            reply.clearStatus(); // Only send status on first chunk.
          }
        }
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

  class ExecutionServer extends ExecuteServiceImplBase {
    private final Path workPath;
    private final RemoteWorkerOptions options;

    public ExecutionServer(Path workPath, RemoteWorkerOptions options) {
      this.workPath = workPath;
      this.options = options;
    }

    private Map<String, String> getEnvironmentVariables(RemoteProtocol.Command command) {
      HashMap<String, String> result = new HashMap<>();
      for (EnvironmentEntry entry : command.getEnvironmentList()) {
        result.put(entry.getVariable(), entry.getValue());
      }
      return result;
    }

    public ExecuteReply execute(Action action, Path execRoot)
        throws IOException, InterruptedException {
      ByteArrayOutputStream stdout = new ByteArrayOutputStream();
      ByteArrayOutputStream stderr = new ByteArrayOutputStream();
      try {
        RemoteProtocol.Command command =
            RemoteProtocol.Command.parseFrom(cache.downloadBlob(action.getCommandDigest()));
        cache.downloadTree(action.getInputRootDigest(), execRoot);

        List<Path> outputs = new ArrayList<>(action.getOutputPathList().size());
        for (String output : action.getOutputPathList()) {
          Path file = execRoot.getRelative(output);
          if (file.exists()) {
            throw new FileAlreadyExistsException("Output file already exists: " + file);
          }
          FileSystemUtils.createDirectoryAndParents(file.getParentDirectory());
          outputs.add(file);
        }

        // TODO(olaola): time out after specified server-side deadline.
        Command cmd =
            new Command(
                command.getArgvList().toArray(new String[] {}),
                getEnvironmentVariables(command),
                new File(execRoot.getPathString()));
        cmd.execute(Command.NO_INPUT, Command.NO_OBSERVER, stdout, stderr, true);

        // Execute throws a CommandException on non-zero return values, so action has succeeded.
        ImmutableList<ContentDigest> outErrDigests =
            cache.uploadBlobs(ImmutableList.of(stdout.toByteArray(), stderr.toByteArray()));
        ActionResult.Builder result =
            ActionResult.newBuilder()
                .setReturnCode(0)
                .setStdoutDigest(outErrDigests.get(0))
                .setStderrDigest(outErrDigests.get(1));
        cache.uploadAllResults(execRoot, outputs, result);
        cache.setCachedActionResult(ContentDigests.computeActionKey(action), result.build());
        return ExecuteReply.newBuilder()
            .setResult(result)
            .setStatus(ExecutionStatus.newBuilder().setExecuted(true).setSucceeded(true))
            .build();
      } catch (CommandException e) {
        ImmutableList<ContentDigest> outErrDigests =
            cache.uploadBlobs(ImmutableList.of(stdout.toByteArray(), stderr.toByteArray()));
        final int returnCode =
            e instanceof AbnormalTerminationException
                ? ((AbnormalTerminationException) e)
                    .getResult()
                    .getTerminationStatus()
                    .getExitCode()
                : -1;
        return ExecuteReply.newBuilder()
            .setResult(
                ActionResult.newBuilder()
                    .setReturnCode(returnCode)
                    .setStdoutDigest(outErrDigests.get(0))
                    .setStderrDigest(outErrDigests.get(1)))
            .setStatus(
                ExecutionStatus.newBuilder()
                    .setExecuted(true)
                    .setSucceeded(false)
                    .setError(ExecutionStatus.ErrorCode.EXEC_FAILED)
                    .setErrorDetail(e.toString()))
            .build();
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
        if (options.debug) {
          if (!reply.getStatus().getSucceeded()) {
            LOG.warning("Work failed. Request: " + request.toString() + ".");
          } else if (LOG_FINER) {
            LOG.fine("Work completed.");
          }
        }
        if (!options.debug) {
          FileSystemUtils.deleteTree(tempRoot);
        } else {
          LOG.warning("Preserving work directory " + tempRoot.toString() + ".");
        }
      } catch (IOException | InterruptedException e) {
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

    if (remoteWorkerOptions.workPath == null) {
      printUsage(parser);
      return;
    }

    System.out.println("*** Initializing in-memory cache server.");
    ConcurrentMap<String, byte[]> cache =
        ConcurrentMapFactory.isRemoteCacheOptions(remoteOptions)
            ? ConcurrentMapFactory.create(remoteOptions)
            : new ConcurrentHashMap<String, byte[]>();

    System.out.println(
        "*** Starting grpc server on all locally bound IPs on port "
            + remoteWorkerOptions.listenPort
            + ".");
    Path workPath = getFileSystem().getPath(remoteWorkerOptions.workPath);
    FileSystemUtils.createDirectoryAndParents(workPath);
    RemoteWorker worker =
        new RemoteWorker(workPath, remoteWorkerOptions, new ConcurrentMapActionCache(cache));
    final Server server =
        ServerBuilder.forPort(remoteWorkerOptions.listenPort)
            .addService(worker.getCasServer())
            .addService(worker.getExecutionServer())
            .addService(worker.getExecCacheServer())
            .build();
    server.start();

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

  public static void printUsage(OptionsParser parser) {
    System.out.println("Usage: remote_worker \n\n" + "Starts a worker that runs a gRPC service.");
    System.out.println(
        parser.describeOptions(
            Collections.<String, String>emptyMap(), OptionsParser.HelpVerbosity.LONG));
  }

  static FileSystem getFileSystem() {
    return OS.getCurrent() == OS.WINDOWS ? new JavaIoFileSystem() : new UnixFileSystem();
  }
}
