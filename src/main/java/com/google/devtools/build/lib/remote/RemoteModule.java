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

package com.google.devtools.build.lib.remote;

import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport.TransportKind;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.remote.blobstore.BlobStoreBuildEventArtifactUploader;
import com.google.devtools.build.lib.remote.blobstore.OnDiskBlobStore;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.remote.blobstore.http.HttpBlobStore;
import com.google.devtools.build.lib.remote.logging.LoggingInterceptor;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.AsynchronousFileOutputStream;
import com.google.devtools.build.lib.vfs.FileSystem.HashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import io.grpc.CallCredentials;
import io.grpc.Channel;
import io.grpc.ClientInterceptors;
import java.io.IOException;
import java.net.URI;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/** RemoteModule provides distributed cache and remote execution for Bazel. */
public final class RemoteModule extends BlazeModule {
  private static final Logger logger = Logger.getLogger(RemoteModule.class.getName());
  private final BuildEventArtifactUploaderDelegate buildEventArtifactUploader =
      new BuildEventArtifactUploaderDelegate();

  private ByteStreamUploader uploader;
  private ListeningScheduledExecutorService retryScheduler;
  private AsynchronousFileOutputStream rpcLogFile;
  private RemoteActionContextProvider actionContextProvider;

  @Override
  public void serverInit(OptionsProvider startupOptions, ServerBuilder builder) {
    builder.addBuildEventArtifactUploader(buildEventArtifactUploader);
  }

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    env.getEventBus().register(this);
    String buildRequestId = env.getBuildRequestId().toString();
    String commandId = env.getCommandId().toString();
    logger.info("Command: buildRequestId = " + buildRequestId + ", commandId = " + commandId);
    Path logDir =
        env.getOutputBase().getRelative(env.getRuntime().getProductName() + "-remote-logs");
    try {
      // Clean out old logs files.
      if (logDir.exists()) {
        FileSystemUtils.deleteTree(logDir);
      }
      logDir.createDirectory();
    } catch (IOException e) {
      env.getReporter()
          .handle(Event.error("Could not create base directory for remote logs: " + logDir));
      throw new AbruptExitException(ExitCode.LOCAL_ENVIRONMENTAL_ERROR, e);
    }
    RemoteOptions options = env.getOptions().getOptions(RemoteOptions.class);
    AuthAndTLSOptions authAndTlsOptions = env.getOptions().getOptions(AuthAndTLSOptions.class);
    HashFunction hashFn = env.getRuntime().getFileSystem().getDigestFunction();
    DigestUtil digestUtil = new DigestUtil(hashFn);

    // Quit if no remote options specified.
    if (options == null) {
      return;
    }

    boolean enableHttpCache = options.remoteHttpCache != null;
    boolean enableDiskCache = options.diskCache != null;
    boolean enableBlobStoreCache = enableHttpCache || enableDiskCache;
    boolean enableGrpcCache = options.remoteCache != null;
    boolean enableExecutor = options.remoteExecutor != null;

    if (enableHttpCache && enableDiskCache) {
      throw new AbruptExitException(
          "Cannot enable HTTP-based and local disk cache simultaneously",
          ExitCode.COMMAND_LINE_ERROR);
    }
    if (enableBlobStoreCache && enableExecutor) {
      throw new AbruptExitException(
          "Cannot combine gRPC based remote execution with local disk or HTTP-based caching",
          ExitCode.COMMAND_LINE_ERROR);
    }

    try {
      LoggingInterceptor logger = null;
      if (!options.experimentalRemoteGrpcLog.isEmpty()) {
        rpcLogFile = new AsynchronousFileOutputStream(options.experimentalRemoteGrpcLog);
        logger = new LoggingInterceptor(rpcLogFile, env.getRuntime().getClock());
      }
      RemoteRetrier retrier =
          new RemoteRetrier(options, RemoteRetrier.RETRIABLE_GRPC_ERRORS, Retrier.ALLOW_ALL_CALLS);

      final AbstractRemoteActionCache cache;
      if (enableHttpCache) {
        URI uri = URI.create(options.remoteHttpCache);
        Credentials credentials = GoogleAuthUtils.newCredentials(authAndTlsOptions);
        int timeoutSeconds = (int) TimeUnit.SECONDS.toMillis(options.remoteTimeout);
        SimpleBlobStore blobStore = new HttpBlobStore(uri, timeoutSeconds, credentials);
        buildEventArtifactUploader.initialize(
            new BlobStoreBuildEventArtifactUploader(
                blobStore, Collections.singletonList(TransportKind.BES_GRPC)));
        cache = new SimpleBlobStoreActionCache(options, blobStore, digestUtil);
      } else if (enableDiskCache) {
        Path cacheDir = env.getWorkingDirectory().getRelative(options.diskCache);
        if (!cacheDir.exists()) {
          cacheDir.createDirectoryAndParents();
        }
        SimpleBlobStore blobStore = new OnDiskBlobStore(cacheDir);
        buildEventArtifactUploader.initialize(
            new BlobStoreBuildEventArtifactUploader(
                blobStore, Collections.singletonList(TransportKind.BEP_FILE)));
        cache = new SimpleBlobStoreActionCache(options, blobStore, digestUtil);
      } else if (enableGrpcCache || enableExecutor) {
        // If a remote executor but no remote cache is specified, assume both at the same target.
        String target = enableGrpcCache ? options.remoteCache : options.remoteExecutor;
        Channel ch = GoogleAuthUtils.newChannel(target, authAndTlsOptions);
        if (logger != null) {
          ch = ClientInterceptors.intercept(ch, logger);
        }
        retryScheduler = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
        CallCredentials creds = GoogleAuthUtils.newCallCredentials(authAndTlsOptions);
        uploader =
            new ByteStreamUploader(
                options.remoteInstanceName,
                ch,
                creds,
                options.remoteTimeout,
                retrier,
                retryScheduler);
        buildEventArtifactUploader.initialize(
            new ByteStreamBuildEventArtifactUploader(uploader, target, options.remoteInstanceName));
        cache = new GrpcRemoteCache(ch, creds, options, retrier, digestUtil, uploader);
      } else {
        cache = null;
      }
      
      final GrpcRemoteExecutor executor;
      if (enableExecutor) {
        Channel ch = GoogleAuthUtils.newChannel(options.remoteExecutor, authAndTlsOptions);
        if (logger != null) {
          ch = ClientInterceptors.intercept(ch, logger);
        }
        executor =
            new GrpcRemoteExecutor(
                ch,
                GoogleAuthUtils.newCallCredentials(authAndTlsOptions),
                options.remoteTimeout,
                retrier);
      } else {
        executor = null;
      }
      actionContextProvider =
          new RemoteActionContextProvider(env, cache, executor, digestUtil, logDir);
    } catch (Exception e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      env.getBlazeModuleEnvironment()
          .exit(
              new AbruptExitException(
                  "Error initializing RemoteModule", ExitCode.COMMAND_LINE_ERROR));
    }
  }

  @Override
  public void afterCommand() {
    if (rpcLogFile != null) {
      try {
        rpcLogFile.close();
      } catch (IOException e) {
        throw new RuntimeException(e);
      } finally {
        rpcLogFile = null;
      }
    }
    if (retryScheduler != null) {
      try {
        retryScheduler.shutdownNow();
      } finally {
        retryScheduler = null;
      }
    }
    if (uploader != null) {
      try {
        uploader.shutdown();
      } finally {
        uploader = null;
      }
    }
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    if (actionContextProvider != null) {
      builder.addActionContextProvider(actionContextProvider);
    }
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(
            RemoteOptions.class, AuthAndTLSOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  private static class BuildEventArtifactUploaderDelegate implements BuildEventArtifactUploader {

    private BuildEventArtifactUploader delegate;

    void initialize(BuildEventArtifactUploader buildEventArtifactUploader) {
      this.delegate = Preconditions.checkNotNull(buildEventArtifactUploader);
    }

    @Override
    public PathConverter upload(Set<Path> files) throws IOException, InterruptedException {
      Preconditions.checkState(
          delegate != null,
          "No BuildEventArtifactUploader has been specified." + " This is a bug in Bazel.");
      return delegate.upload(files);
    }

    @Override
    public List<TransportKind> supportedTransports() {
      if (delegate == null) {
        return Collections.emptyList();
      }
      return delegate.supportedTransports();
    }
  }
}
