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

import static java.util.logging.Level.FINE;
import static java.util.logging.Level.INFO;

import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.devtools.build.lib.remote.RemoteOptions;
import com.google.devtools.build.lib.remote.SimpleBlobStore;
import com.google.devtools.build.lib.remote.SimpleBlobStoreActionCache;
import com.google.devtools.build.lib.remote.SimpleBlobStoreFactory;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.ProcessUtils;
import com.google.devtools.build.lib.util.SingleLineFormatter;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.remoteexecution.v1test.ActionCacheGrpc.ActionCacheImplBase;
import com.google.devtools.remoteexecution.v1test.ContentAddressableStorageGrpc.ContentAddressableStorageImplBase;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.ExecutionGrpc.ExecutionImplBase;
import com.google.watcher.v1.WatcherGrpc.WatcherImplBase;
import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Implements a remote worker that accepts work items as protobufs. The server implementation is
 * based on gRPC.
 */
public final class RemoteWorker {
  private static final Logger logger = Logger.getLogger(RemoteWorker.class.getName());

  private final RemoteWorkerOptions workerOptions;
  private final ActionCacheImplBase actionCacheServer;
  private final ByteStreamImplBase bsServer;
  private final ContentAddressableStorageImplBase casServer;
  private final WatcherImplBase watchServer;
  private final ExecutionImplBase execServer;

  static FileSystem getFileSystem() {
    return OS.getCurrent() == OS.WINDOWS ? new JavaIoFileSystem() : new UnixFileSystem();
  }

  public RemoteWorker(RemoteWorkerOptions workerOptions, SimpleBlobStoreActionCache cache)
      throws IOException {
    this.workerOptions = workerOptions;
    this.actionCacheServer = new ActionCacheServer(cache);
    this.bsServer = new ByteStreamServer(cache);
    this.casServer = new CasServer(cache);

    if (workerOptions.workPath != null) {
      ConcurrentHashMap<String, ExecuteRequest> operationsCache = new ConcurrentHashMap<>();
      Path workPath = getFileSystem().getPath(workerOptions.workPath);
      FileSystemUtils.createDirectoryAndParents(workPath);
      watchServer = new WatcherServer(workPath, cache, workerOptions, operationsCache);
      execServer = new ExecutionServer(operationsCache);
    } else {
      watchServer = null;
      execServer = null;
    }
  }

  public Server startServer() throws IOException {
    NettyServerBuilder b =
        NettyServerBuilder.forPort(workerOptions.listenPort)
            .addService(actionCacheServer)
            .addService(bsServer)
            .addService(casServer);

    if (execServer != null) {
      b.addService(execServer);
      b.addService(watchServer);
    } else {
      logger.info("Execution disabled, only serving cache requests.");
    }

    Server server = b.build();
    logger.log(INFO, "Starting gRPC server on port {0,number,#}.", workerOptions.listenPort);
    server.start();

    return server;
  }

  private void createPidFile() throws IOException {
    if (workerOptions.pidFile == null) {
      return;
    }

    final Path pidFile = getFileSystem().getPath(workerOptions.pidFile);
    try (PrintWriter printWriter = new PrintWriter(pidFile.getPathFile())) {
      printWriter.println(ProcessUtils.getpid());
    }

    Runtime.getRuntime()
        .addShutdownHook(
            new Thread() {
              @Override
              public void run() {
                try {
                  pidFile.delete();
                } catch (IOException e) {
                  System.err.println("Cannot remove pid file: " + pidFile);
                }
              }
            });
  }

  public static void main(String[] args) throws Exception {
    OptionsParser parser =
        OptionsParser.newOptionsParser(RemoteOptions.class, RemoteWorkerOptions.class);
    parser.parseAndExitUponError(args);
    RemoteOptions remoteOptions = parser.getOptions(RemoteOptions.class);
    RemoteWorkerOptions remoteWorkerOptions = parser.getOptions(RemoteWorkerOptions.class);

    Logger rootLog = Logger.getLogger("");
    rootLog.getHandlers()[0].setFormatter(new SingleLineFormatter());
    if (remoteWorkerOptions.debug) {
      rootLog.setLevel(FINE);
      rootLog.getHandlers()[0].setLevel(FINE);
    }

    logger.info("Initializing in-memory cache server.");
    boolean usingRemoteCache = SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions);
    if (!usingRemoteCache) {
      logger.warning("Not using remote cache. This should be used for testing only!");
    }

    SimpleBlobStore blobStore =
        usingRemoteCache
            ? SimpleBlobStoreFactory.create(remoteOptions)
            : new SimpleBlobStoreFactory.ConcurrentMapBlobStore(
                new ConcurrentHashMap<String, byte[]>());

    RemoteWorker worker =
        new RemoteWorker(remoteWorkerOptions, new SimpleBlobStoreActionCache(blobStore));

    final Server server = worker.startServer();
    worker.createPidFile();
    server.awaitTermination();
  }
}
