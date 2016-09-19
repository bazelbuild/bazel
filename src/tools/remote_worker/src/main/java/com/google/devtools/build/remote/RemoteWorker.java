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
import com.google.devtools.build.lib.remote.ConcurrentMapActionCache;
import com.google.devtools.build.lib.remote.ConcurrentMapFactory;
import com.google.devtools.build.lib.remote.ContentDigests;
import com.google.devtools.build.lib.remote.ExecuteServiceGrpc.ExecuteServiceImplBase;
import com.google.devtools.build.lib.remote.RemoteOptions;
import com.google.devtools.build.lib.remote.RemoteProtocol;
import com.google.devtools.build.lib.remote.RemoteProtocol.Action;
import com.google.devtools.build.lib.remote.RemoteProtocol.ActionResult;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasStatus;
import com.google.devtools.build.lib.remote.RemoteProtocol.Command.EnvironmentEntry;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionStatus;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.ProcessUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.UnixFileSystem;
import com.google.devtools.common.options.OptionsParser;
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
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Implements a remote worker that accepts work items as protobufs. The server implementation is
 * based on grpc.
 */
public class RemoteWorker extends ExecuteServiceImplBase {
  private static final Logger LOG = Logger.getLogger(RemoteWorker.class.getName());
  private static final boolean LOG_FINER = LOG.isLoggable(Level.FINER);
  private final Path workPath;
  private final RemoteOptions remoteOptions;
  private final RemoteWorkerOptions options;
  private final ConcurrentMapActionCache cache;

  public RemoteWorker(
      Path workPath,
      RemoteOptions remoteOptions,
      RemoteWorkerOptions options,
      ConcurrentMapActionCache cache) {
    this.workPath = workPath;
    this.remoteOptions = remoteOptions;
    this.options = options;
    this.cache = cache;
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
              ? ((AbnormalTerminationException) e).getResult().getTerminationStatus().getExitCode()
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

    System.out.println("*** Starting Hazelcast server.");
    ConcurrentMapActionCache cache =
        new ConcurrentMapActionCache(ConcurrentMapFactory.createHazelcast(remoteOptions));

    System.out.println(
        "*** Starting grpc server on all locally bound IPs on port "
            + remoteWorkerOptions.listenPort
            + ".");
    Path workPath = getFileSystem().getPath(remoteWorkerOptions.workPath);
    FileSystemUtils.createDirectoryAndParents(workPath);
    RemoteWorker worker = new RemoteWorker(workPath, remoteOptions, remoteWorkerOptions, cache);
    final Server server =
        ServerBuilder.forPort(remoteWorkerOptions.listenPort).addService(worker).build();
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
    System.out.println("Usage: remote_worker \n\n" + "Starts a worker that runs a RPC service.");
    System.out.println(
        parser.describeOptions(
            Collections.<String, String>emptyMap(), OptionsParser.HelpVerbosity.LONG));
  }

  static FileSystem getFileSystem() {
    return OS.getCurrent() == OS.WINDOWS ? new JavaIoFileSystem() : new UnixFileSystem();
  }
}
