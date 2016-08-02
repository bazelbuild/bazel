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

import com.google.devtools.build.lib.remote.ConcurrentMapActionCache;
import com.google.devtools.build.lib.remote.HazelcastCacheFactory;
import com.google.devtools.build.lib.remote.MemcacheWorkExecutor;
import com.google.devtools.build.lib.remote.RemoteOptions;
import com.google.devtools.build.lib.remote.RemoteProtocol.RemoteWorkRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.RemoteWorkResponse;
import com.google.devtools.build.lib.remote.RemoteWorkGrpc.RemoteWorkImplBase;
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
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.UUID;
import java.util.concurrent.ConcurrentMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Implements a remote worker that accepts work items as protobufs. The server implementation is
 * based on grpc.
 */
public class RemoteWorker extends RemoteWorkImplBase {
  private static final Logger LOG = Logger.getLogger(RemoteWorker.class.getName());
  private static final boolean LOG_FINER = LOG.isLoggable(Level.FINER);
  private final Path workPath;
  private final RemoteOptions remoteOptions;
  private final RemoteWorkerOptions options;
  private final ConcurrentMap<String, byte[]> cache;

  public RemoteWorker(
      Path workPath,
      RemoteOptions remoteOptions,
      RemoteWorkerOptions options,
      ConcurrentMap<String, byte[]> cache) {
    this.workPath = workPath;
    this.remoteOptions = remoteOptions;
    this.options = options;
    this.cache = cache;
  }

  @Override
  public void executeSynchronously(
      RemoteWorkRequest request, StreamObserver<RemoteWorkResponse> responseObserver) {
    Path tempRoot = workPath.getRelative("build-" + UUID.randomUUID().toString());
    try {
      FileSystemUtils.createDirectoryAndParents(tempRoot);
      final ConcurrentMapActionCache actionCache =
          new ConcurrentMapActionCache(tempRoot, remoteOptions, cache);
      final MemcacheWorkExecutor workExecutor =
          MemcacheWorkExecutor.createLocalWorkExecutor(actionCache, tempRoot);
      if (LOG_FINER) {
        LOG.fine(
            "Work received has "
                + request.getInputFilesCount()
                + " input files and "
                + request.getOutputFilesCount()
                + " output files.");
      }
      RemoteWorkResponse response = workExecutor.executeLocally(request);
      responseObserver.onNext(response);
      if (options.debug) {
        if (!response.getSuccess()) {
          LOG.warning("Work failed. Request: " + request.toString() + ".");

        } else if (LOG_FINER) {
          LOG.fine("Work completed.");
        }
      }
      if (!options.debug || response.getSuccess()) {
        FileSystemUtils.deleteTree(tempRoot);
      } else {
        LOG.warning("Preserving work directory " + tempRoot.toString() + ".");
      }
    } catch (IOException e) {
      RemoteWorkResponse.Builder response = RemoteWorkResponse.newBuilder();
      response.setSuccess(false).setOut("").setErr("").setException(e.toString());
      responseObserver.onNext(response.build());
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
    ConcurrentMap<String, byte[]> cache = new HazelcastCacheFactory().create(remoteOptions);

    System.out.println("*** Starting grpc server on all locally bound IPs on port "
        + remoteWorkerOptions.listenPort + ".");
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
