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

import build.bazel.remote.execution.v2.DigestFunction;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.LocalFilesArtifactUploader;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.remote.logging.LoggingInterceptor;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOptions.FetchRemoteOutputsStrategy;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BuildEventArtifactUploaderFactory;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.AsynchronousFileOutputStream;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.rpc.PreconditionFailure;
import com.google.rpc.PreconditionFailure.Violation;
import io.grpc.CallCredentials;
import io.grpc.ClientInterceptor;
import io.grpc.Context;
import io.grpc.Status.Code;
import io.grpc.protobuf.StatusProto;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.function.Predicate;

/** RemoteModule provides distributed cache and remote execution for Bazel. */
public final class RemoteModule extends BlazeModule {

  private AsynchronousFileOutputStream rpcLogFile;

  private final ListeningScheduledExecutorService retryScheduler =
      MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  private RemoteActionContextProvider actionContextProvider;

  private final BuildEventArtifactUploaderFactoryDelegate
      buildEventArtifactUploaderFactoryDelegate = new BuildEventArtifactUploaderFactoryDelegate();

  @Override
  public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder) {
    builder.addBuildEventArtifactUploaderFactory(buildEventArtifactUploaderFactoryDelegate, "remote");
  }

  private static final String VIOLATION_TYPE_MISSING = "MISSING";

  private static final Predicate<? super Exception> RETRIABLE_EXEC_ERRORS =
      e -> {
        if (e instanceof CacheNotFoundException || e.getCause() instanceof CacheNotFoundException) {
          return true;
        }
        if (!RemoteRetrierUtils.causedByStatus(e, Code.FAILED_PRECONDITION)) {
          return false;
        }
        com.google.rpc.Status status = StatusProto.fromThrowable(e);
        if (status == null || status.getDetailsCount() == 0) {
          return false;
        }
        for (Any details : status.getDetailsList()) {
          PreconditionFailure f;
          try {
            f = details.unpack(PreconditionFailure.class);
          } catch (InvalidProtocolBufferException protoEx) {
            return false;
          }
          if (f.getViolationsCount() == 0) {
            return false; // Generally shouldn't happen
          }
          for (Violation v : f.getViolationsList()) {
            if (!v.getType().equals(VIOLATION_TYPE_MISSING)) {
              return false;
            }
          }
        }
        return true; // if *all* > 0 violations have type MISSING
      };

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    RemoteOptions remoteOptions = env.getOptions().getOptions(RemoteOptions.class);
    if (!remoteEnabled(remoteOptions)) {
      return;
    }

    env.getEventBus().register(this);

    String invocationId = env.getCommandId().toString();
    String buildRequestId = env.getBuildRequestId();
    env.getReporter().handle(Event.info(String.format("Invocation ID: %s", invocationId)));

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
    AuthAndTLSOptions authAndTlsOptions = env.getOptions().getOptions(AuthAndTLSOptions.class);
    DigestHashFunction hashFn = env.getRuntime().getFileSystem().getDigestFunction();
    DigestUtil digestUtil = new DigestUtil(hashFn);

    boolean enableRestCache = SimpleBlobStoreFactory.isRestUrlOptions(remoteOptions);
    boolean enableDiskCache = SimpleBlobStoreFactory.isDiskCache(remoteOptions);
    if (enableRestCache && enableDiskCache) {
      throw new AbruptExitException(
          "Cannot enable HTTP-based and local disk cache simultaneously",
          ExitCode.COMMAND_LINE_ERROR);
    }
    boolean enableBlobStoreCache = enableRestCache || enableDiskCache;
    boolean enableGrpcCache = GrpcRemoteCache.isRemoteCacheOptions(remoteOptions);
    if (enableBlobStoreCache && !Strings.isNullOrEmpty(remoteOptions.remoteExecutor)) {
      throw new AbruptExitException(
          "Cannot combine gRPC based remote execution with local disk or HTTP-based caching",
          ExitCode.COMMAND_LINE_ERROR);
    }

    try {
      List<ClientInterceptor> interceptors = new ArrayList<>();
      if (!remoteOptions.experimentalRemoteGrpcLog.isEmpty()) {
        rpcLogFile = new AsynchronousFileOutputStream(remoteOptions.experimentalRemoteGrpcLog);
        interceptors.add(new LoggingInterceptor(rpcLogFile, env.getRuntime().getClock()));
      }

      ReferenceCountedChannel cacheChannel = null;
      ReferenceCountedChannel execChannel = null;
      RemoteRetrier rpcRetrier = null;
      // Initialize the gRPC channels and capabilities service, when relevant.
      if (!Strings.isNullOrEmpty(remoteOptions.remoteExecutor)) {
        execChannel =
            new ReferenceCountedChannel(
                GoogleAuthUtils.newChannel(
                    remoteOptions.remoteExecutor,
                    authAndTlsOptions,
                    interceptors.toArray(new ClientInterceptor[0])));
      }
      RemoteRetrier executeRetrier = null;
      AbstractRemoteActionCache cache = null;
      if (enableGrpcCache || !Strings.isNullOrEmpty(remoteOptions.remoteExecutor)) {
        rpcRetrier =
              new RemoteRetrier(
                  remoteOptions,
                  RemoteRetrier.RETRIABLE_GRPC_ERRORS,
                  retryScheduler,
                  Retrier.ALLOW_ALL_CALLS);
        if (!Strings.isNullOrEmpty(remoteOptions.remoteCache)
            && !remoteOptions.remoteCache.equals(remoteOptions.remoteExecutor)) {
          cacheChannel =
              new ReferenceCountedChannel(
                  GoogleAuthUtils.newChannel(
                      remoteOptions.remoteCache,
                      authAndTlsOptions,
                      interceptors.toArray(new ClientInterceptor[0])));
        } else {  // Assume --remote_cache is equal to --remote_executor by default.
          cacheChannel = execChannel.retain(); // execChannel is guaranteed to be defined here.
        }
        CallCredentials credentials = GoogleAuthUtils.newCallCredentials(authAndTlsOptions);
        // We always query the execution server for capabilities, if it is defined. A remote
        // execution/cache system should have all its servers to return the capabilities pertaining
        // to the system as a whole.
        RemoteServerCapabilities rsc = new RemoteServerCapabilities(
                remoteOptions.remoteInstanceName,
                (execChannel != null ? execChannel : cacheChannel),
                credentials,
                remoteOptions.remoteTimeout,
                rpcRetrier);
        ServerCapabilities capabilities = null;
        try {
          capabilities = rsc.get(buildRequestId, invocationId);
        } catch (IOException e) {
          throw new AbruptExitException(
              "Failed to query remote execution capabilities: " + e.getMessage(),
              ExitCode.REMOTE_ERROR,
              e);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          return;
        }
        checkClientServerCompatibility(
            capabilities, remoteOptions, digestUtil.getDigestFunction(), env.getReporter());
        executeRetrier = createExecuteRetrier(remoteOptions, retryScheduler);
        ByteStreamUploader uploader =
            new ByteStreamUploader(
                remoteOptions.remoteInstanceName,
                cacheChannel.retain(),
                credentials,
                remoteOptions.remoteTimeout,
                rpcRetrier);
        cacheChannel.release();
        cache =
            new GrpcRemoteCache(
                cacheChannel.retain(),
                credentials,
                uploader.retain(),
                remoteOptions,
                digestUtil,
                rpcRetrier);
        uploader.release();
        // Context requestContext =
        //     TracingMetadataUtils.contextWithMetadata(buildRequestId, invocationId, "bes-upload");
        // buildEventArtifactUploaderFactoryDelegate.init(
        //     new ByteStreamBuildEventArtifactUploaderFactory(
        //         uploader,
        //         cacheChannel.authority(),
        //         requestContext,
        //         remoteOptions.remoteInstanceName));
      }

      if (enableBlobStoreCache) {
        Retrier retrier =
            new Retrier(
                () -> Retrier.RETRIES_DISABLED,
                (e) -> false,
                retryScheduler,
                Retrier.ALLOW_ALL_CALLS);
        executeRetrier = null;
        cache =
            new SimpleBlobStoreActionCache(
                SimpleBlobStoreFactory.create(
                    remoteOptions,
                    GoogleAuthUtils.newCredentials(authAndTlsOptions),
                    env.getWorkingDirectory()),
                remoteOptions,
                retrier,
                digestUtil);
      }

      GrpcRemoteExecutor executor = null;
      if (!Strings.isNullOrEmpty(remoteOptions.remoteExecutor)) {
        RemoteRetrier retrier =
            new RemoteRetrier(
                remoteOptions,
                RemoteRetrier.RETRIABLE_GRPC_EXEC_ERRORS,
                retryScheduler,
                Retrier.ALLOW_ALL_CALLS);
        executor =
            new GrpcRemoteExecutor(
                execChannel.retain(),
                GoogleAuthUtils.newCallCredentials(authAndTlsOptions),
                retrier);
        execChannel.release();
      }
      actionContextProvider =
          new RemoteActionContextProvider(env, cache, executor, executeRetrier, digestUtil, logDir);
    } catch (IOException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      env.getBlazeModuleEnvironment()
          .exit(
              new AbruptExitException(
                  "Error initializing RemoteModule", ExitCode.COMMAND_LINE_ERROR));
    }
  }

  private void checkClientServerCompatibility(
      ServerCapabilities capabilities,
      RemoteOptions remoteOptions,
      DigestFunction digestFunction,
      Reporter reporter)
      throws AbruptExitException {
    RemoteServerCapabilities.ClientServerCompatibilityStatus st =
        RemoteServerCapabilities.checkClientServerCompatibility(
            capabilities, remoteOptions, digestFunction);
    for (String warning : st.getWarnings()) {
      reporter.handle(Event.warn(warning));
    }
    List<String> errors = st.getErrors();
    for (int i = 0; i < errors.size() - 1; ++i) {
      reporter.handle(Event.error(errors.get(i)));
    }
    if (!errors.isEmpty()) {
      throw new AbruptExitException(errors.get(errors.size() - 1), ExitCode.REMOTE_ERROR);
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
    buildEventArtifactUploaderFactoryDelegate.reset();
  }

  private boolean remoteEnabled(RemoteOptions options) {
    return options != null && (!Strings.isNullOrEmpty(options.remoteHttpCache) ||
        !Strings.isNullOrEmpty(options.remoteCache) ||
        !Strings.isNullOrEmpty(options.remoteExecutor));
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    RemoteOptions remoteOptions = env.getOptions().getOptions(RemoteOptions.class);
    if (!remoteEnabled(remoteOptions)) {
      return;
    }

    RemoteActionContextProvider actionContextProvider = this.actionContextProvider;
    if (actionContextProvider == null) {
      return;
    }
    builder.addActionContextProvider(actionContextProvider);

    FetchRemoteOutputsStrategy remoteOutputsStrategy = remoteOptions.experimentalRemoteFetchOutputs;
    switch (remoteOutputsStrategy) {
      case ALL:
        break;
      case MINIMAL:
        Context ctx = TracingMetadataUtils.contextWithMetadata(env.getBuildRequestId(),
            env.getCommandId().toString(), "prefetch-inputs");
        builder.setActionInputPrefetcher(
            new RemoteActionInputFetcher(actionContextProvider.getRemoteCache(), env.getExecRoot(), ctx));
        break;
    }
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(RemoteOptions.class, AuthAndTLSOptions.class)
        : ImmutableList.of();
  }

  static RemoteRetrier createExecuteRetrier(
      RemoteOptions options, ListeningScheduledExecutorService retryService) {
    return new RemoteRetrier(
        options.experimentalRemoteRetry
            ? () -> new Retrier.ZeroBackoff(options.experimentalRemoteRetryMaxAttempts)
            : () -> Retrier.RETRIES_DISABLED,
        RemoteModule.RETRIABLE_EXEC_ERRORS,
        retryService,
        Retrier.ALLOW_ALL_CALLS);
  }

  private static class BuildEventArtifactUploaderFactoryDelegate
      implements BuildEventArtifactUploaderFactory {

    private volatile BuildEventArtifactUploaderFactory uploaderFactory;

    public void init(BuildEventArtifactUploaderFactory uploaderFactory) {
      Preconditions.checkState(this.uploaderFactory == null);
      this.uploaderFactory = uploaderFactory;
    }

    public void reset() {
      this.uploaderFactory = null;
    }

    @Override
    public BuildEventArtifactUploader create(CommandEnvironment env) {
      BuildEventArtifactUploaderFactory uploaderFactory0 = this.uploaderFactory;
      if (uploaderFactory0 == null) {
        return new LocalFilesArtifactUploader();
      }
      return uploaderFactory0.create(env);
    }
  }
}
