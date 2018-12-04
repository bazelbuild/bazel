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

package com.google.devtools.build.remote.worker;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.logging.Level.FINE;
import static java.util.logging.Level.INFO;
import static java.util.logging.Level.SEVERE;

import build.bazel.remote.execution.v2.ActionCacheGrpc.ActionCacheImplBase;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.CapabilitiesGrpc.CapabilitiesImplBase;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc.ContentAddressableStorageImplBase;
import build.bazel.remote.execution.v2.ExecutionGrpc.ExecutionImplBase;
import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.RemoteRetrier;
import com.google.devtools.build.lib.remote.Retrier;
import com.google.devtools.build.lib.remote.SimpleBlobStoreActionCache;
import com.google.devtools.build.lib.remote.SimpleBlobStoreFactory;
import com.google.devtools.build.lib.remote.blobstore.ConcurrentMapBlobStore;
import com.google.devtools.build.lib.remote.blobstore.OnDiskBlobStore;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.sandbox.LinuxSandboxUtil;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.ProcessUtils;
import com.google.devtools.build.lib.util.SingleLineFormatter;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.DigestHashFunction.DigestFunctionConverter;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import io.grpc.Server;
import io.grpc.ServerInterceptor;
import io.grpc.ServerInterceptors;
import io.grpc.netty.NettyServerBuilder;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.Channel;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.logging.LogLevel;
import io.netty.handler.logging.LoggingHandler;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Implements a remote worker that accepts work items as protobufs. The server implementation is
 * based on gRPC.
 */
public final class RemoteWorker {

  // We need to keep references to the root and netty loggers to prevent them from being garbage
  // collected, which would cause us to loose their configuration.
  private static final Logger rootLogger = Logger.getLogger("");
  private static final Logger nettyLogger = Logger.getLogger("io.grpc.netty");
  private static final Logger logger = Logger.getLogger(RemoteWorker.class.getName());

  private final RemoteWorkerOptions workerOptions;
  private final ActionCacheImplBase actionCacheServer;
  private final ByteStreamImplBase bsServer;
  private final ContentAddressableStorageImplBase casServer;
  private final ExecutionImplBase execServer;
  private final CapabilitiesImplBase capabilitiesServer;

  static FileSystem getFileSystem() {
    final DigestHashFunction hashFunction;
    String value = null;
    try {
      value = System.getProperty("bazel.DigestFunction", "SHA256");
      hashFunction = new DigestFunctionConverter().convert(value);
    } catch (OptionsParsingException e) {
      throw new Error("The specified hash function '" + value + "' is not supported.");
    }
    return new JavaIoFileSystem(hashFunction);
  }

  public RemoteWorker(
      FileSystem fs,
      RemoteWorkerOptions workerOptions,
      SimpleBlobStoreActionCache cache,
      Path sandboxPath,
      DigestUtil digestUtil)
      throws IOException {
    this.workerOptions = workerOptions;
    this.actionCacheServer = new ActionCacheServer(cache, digestUtil);
    Path workPath;
    if (workerOptions.workPath != null) {
      workPath = fs.getPath(workerOptions.workPath);
    } else {
      // TODO(ulfjack): The plan is to make the on-disk storage the default, so we always need to
      // provide a path to the remote worker, and we can then also use that as the work path. E.g.:
      // /given/path/cas/
      // /given/path/upload/
      // /given/path/work/
      // We could technically use a different path for temporary files and execution, but we want
      // the cas/ directory to be on the same file system as the upload/ and work/ directories so
      // that we can atomically move files between them, and / or use hard-links for the exec
      // directories.
      // For now, we use a temporary path if no work path was provided.
      workPath = fs.getPath("/tmp/remote-worker");
    }
    this.bsServer = new ByteStreamServer(cache, workPath, digestUtil);
    this.casServer = new CasServer(cache);

    if (workerOptions.workPath != null) {
      ConcurrentHashMap<String, ListenableFuture<ActionResult>> operationsCache =
          new ConcurrentHashMap<>();
      FileSystemUtils.createDirectoryAndParents(workPath);
      execServer =
          new ExecutionServer(
              workPath, sandboxPath, workerOptions, cache, operationsCache, digestUtil);
    } else {
      execServer = null;
    }
    this.capabilitiesServer = new CapabilitiesServer(digestUtil, execServer != null);
  }

  public Server startServer() throws IOException {
    ServerInterceptor headersInterceptor = new TracingMetadataUtils.ServerHeadersInterceptor();
    NettyServerBuilder b =
        NettyServerBuilder.forPort(workerOptions.listenPort)
            .addService(ServerInterceptors.intercept(actionCacheServer, headersInterceptor))
            .addService(ServerInterceptors.intercept(bsServer, headersInterceptor))
            .addService(ServerInterceptors.intercept(casServer, headersInterceptor))
            .addService(ServerInterceptors.intercept(capabilitiesServer, headersInterceptor));

    if (execServer != null) {
      b.addService(ServerInterceptors.intercept(execServer, headersInterceptor));
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
    try (Writer writer =
        new OutputStreamWriter(pidFile.getOutputStream(), StandardCharsets.UTF_8)) {
      writer.write(Integer.toString(ProcessUtils.getpid()));
      writer.write("\n");
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

  @SuppressWarnings("FutureReturnValueIgnored")
  public static void main(String[] args) throws Exception {
    OptionsParser parser =
        OptionsParser.newOptionsParser(RemoteOptions.class, RemoteWorkerOptions.class);
    parser.parseAndExitUponError(args);
    RemoteOptions remoteOptions = parser.getOptions(RemoteOptions.class);
    RemoteWorkerOptions remoteWorkerOptions = parser.getOptions(RemoteWorkerOptions.class);

    rootLogger.getHandlers()[0].setFormatter(new SingleLineFormatter());
    if (remoteWorkerOptions.debug) {
      rootLogger.getHandlers()[0].setLevel(FINE);
    }

    // Only log severe log messages from Netty. Otherwise it logs warnings that look like this:
    //
    // 170714 08:16:28.552:WT 18 [io.grpc.netty.NettyServerHandler.onStreamError] Stream Error
    // io.netty.handler.codec.http2.Http2Exception$StreamException: Received DATA frame for an
    // unknown stream 11369
    //
    // As far as we can tell, these do not indicate any problem with the connection. We believe they
    // happen when the local side closes a stream, but the remote side hasn't received that
    // notification yet, so there may still be packets for that stream en-route to the local
    // machine. The wording 'unknown stream' is misleading - the stream was previously known, but
    // was recently closed. I'm told upstream discussed this, but didn't want to keep information
    // about closed streams around.
    nettyLogger.setLevel(Level.SEVERE);

    FileSystem fs = getFileSystem();
    Path sandboxPath = null;
    if (remoteWorkerOptions.sandboxing) {
      sandboxPath = prepareSandboxRunner(fs, remoteWorkerOptions);
    }

    logger.info("Initializing in-memory cache server.");
    boolean usingRemoteCache = SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions);
    if (!usingRemoteCache) {
      logger.warning("Not using remote cache. This should be used for testing only!");
    }
    if ((remoteWorkerOptions.casPath != null)
        && (!PathFragment.create(remoteWorkerOptions.casPath).isAbsolute()
            || !fs.getPath(remoteWorkerOptions.casPath).exists())) {
      logger.severe("--cas_path must refer to an existing, absolute path!");
      System.exit(1);
      return;
    }

    // The instance of SimpleBlobStore used is based on these criteria in order:
    // 1. If remote cache or local disk cache is specified then use it first.
    // 2. Finally use a ConcurrentMap to back the blob store.
    final SimpleBlobStore blobStore;
    if (usingRemoteCache) {
      blobStore = SimpleBlobStoreFactory.create(remoteOptions, null, null);
    } else if (remoteWorkerOptions.casPath != null) {
      blobStore = new OnDiskBlobStore(fs.getPath(remoteWorkerOptions.casPath));
    } else {
      blobStore = new ConcurrentMapBlobStore(new ConcurrentHashMap<>());
    }

    ListeningScheduledExecutorService retryService =
        MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));

    RemoteRetrier retrier =
        new RemoteRetrier(
            remoteOptions,
            RemoteRetrier.RETRIABLE_GRPC_ERRORS,
            retryService,
            Retrier.ALLOW_ALL_CALLS);
    DigestUtil digestUtil = new DigestUtil(fs.getDigestFunction());
    RemoteWorker worker =
        new RemoteWorker(
            fs,
            remoteWorkerOptions,
            new SimpleBlobStoreActionCache(blobStore, remoteOptions, retrier, digestUtil),
            sandboxPath,
            digestUtil);

    final Server server = worker.startServer();

    EventLoopGroup bossGroup = null, workerGroup = null;
    Channel ch = null;
    if (remoteWorkerOptions.httpListenPort != 0) {
      // Configure the server.
      bossGroup = new NioEventLoopGroup(1);
      workerGroup = new NioEventLoopGroup();
      ServerBootstrap b = new ServerBootstrap();
      b.group(bossGroup, workerGroup)
          .channel(NioServerSocketChannel.class)
          .handler(new LoggingHandler(LogLevel.INFO))
          .childHandler(new HttpCacheServerInitializer());
      ch = b.bind(remoteWorkerOptions.httpListenPort).sync().channel();
      logger.log(
          INFO,
          "Started HTTP cache server on port " + remoteWorkerOptions.httpListenPort);
    } else {
      logger.log(INFO, "Not starting HTTP cache server");
    }

    worker.createPidFile();

    server.awaitTermination();
    if (ch != null) {
      ch.closeFuture().sync().get();
    }

    retryService.shutdownNow();
    if (bossGroup != null) {
      bossGroup.shutdownGracefully();
    }
    if (workerGroup != null) {
      workerGroup.shutdownGracefully();
    }
  }

  private static Path prepareSandboxRunner(FileSystem fs, RemoteWorkerOptions remoteWorkerOptions) {
    if (OS.getCurrent() != OS.LINUX) {
      logger.severe("Sandboxing requested, but it is currently only available on Linux.");
      System.exit(1);
    }

    if (remoteWorkerOptions.workPath == null) {
      logger.severe("Sandboxing requested, but --work_path was not specified.");
      System.exit(1);
    }

    InputStream sandbox = RemoteWorker.class.getResourceAsStream("/main/tools/linux-sandbox");
    if (sandbox == null) {
      logger.severe(
          "Sandboxing requested, but could not find bundled linux-sandbox binary. "
              + "Please rebuild a worker_deploy.jar on Linux to make this work.");
      System.exit(1);
    }

    Path sandboxPath = null;
    try {
      sandboxPath = fs.getPath(remoteWorkerOptions.workPath).getChild("linux-sandbox");
      try (FileOutputStream fos = new FileOutputStream(sandboxPath.getPathString())) {
        ByteStreams.copy(sandbox, fos);
      }
      sandboxPath.setExecutable(true);
    } catch (IOException e) {
      logger.log(SEVERE, "Could not extract the bundled linux-sandbox binary to " + sandboxPath, e);
      System.exit(1);
    }

    CommandResult cmdResult = null;
    Command cmd =
        new Command(
            LinuxSandboxUtil.commandLineBuilder(sandboxPath, ImmutableList.of("true"))
                .build()
                .toArray(new String[0]),
            ImmutableMap.of(),
            sandboxPath.getParentDirectory().getPathFile());
    try {
      cmdResult = cmd.execute();
    } catch (CommandException e) {
      logger.log(
          SEVERE,
          "Sandboxing requested, but it failed to execute 'true' as a self-check: "
              + new String(cmdResult.getStderr(), UTF_8),
          e);
      System.exit(1);
    }

    return sandboxPath;
  }
}
