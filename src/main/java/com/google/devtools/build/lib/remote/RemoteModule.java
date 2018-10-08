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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploaderFactory;
import com.google.devtools.build.lib.buildeventstream.LocalFilesArtifactUploader;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.remote.Retrier.RetryException;
import com.google.devtools.build.lib.remote.logging.LoggingInterceptor;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.runtime.BlazeModule;
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
import io.grpc.ManagedChannel;
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
        if (!(e instanceof RetryException)
            || !RemoteRetrierUtils.causedByStatus((RetryException) e, Code.FAILED_PRECONDITION)) {
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
    RemoteOptions remoteOptions = env.getOptions().getOptions(RemoteOptions.class);
    AuthAndTLSOptions authAndTlsOptions = env.getOptions().getOptions(AuthAndTLSOptions.class);
    DigestHashFunction hashFn = env.getRuntime().getFileSystem().getDigestFunction();
    DigestUtil digestUtil = new DigestUtil(hashFn);

    // Quit if no remote options specified.
    if (remoteOptions == null) {
      return;
    }

    boolean enableRestCache = SimpleBlobStoreFactory.isRestUrlOptions(remoteOptions);
    boolean enableDiskCache = SimpleBlobStoreFactory.isDiskCache(remoteOptions);
    if (enableRestCache && enableDiskCache) {
      throw new AbruptExitException(
          "Cannot enable HTTP-based and local disk cache simultaneously",
          ExitCode.COMMAND_LINE_ERROR);
    }
    boolean enableBlobStoreCache = enableRestCache || enableDiskCache;
    boolean enableGrpcCache = GrpcRemoteCache.isRemoteCacheOptions(remoteOptions);
    if (enableBlobStoreCache && remoteOptions.remoteExecutor != null) {
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

      final RemoteRetrier executeRetrier;
      final AbstractRemoteActionCache cache;
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
                remoteOptions,
                SimpleBlobStoreFactory.create(
                    remoteOptions,
                    GoogleAuthUtils.newCredentials(authAndTlsOptions),
                    env.getWorkingDirectory()),
                retrier,
                digestUtil);
      } else if (enableGrpcCache || remoteOptions.remoteExecutor != null) {
        // If a remote executor but no remote cache is specified, assume both at the same target.
        String target = enableGrpcCache ? remoteOptions.remoteCache : remoteOptions.remoteExecutor;
        ReferenceCountedChannel channel =
            new ReferenceCountedChannel(
                GoogleAuthUtils.newChannel(
                    target,
                    authAndTlsOptions,
                    interceptors.toArray(new ClientInterceptor[0])));
        RemoteRetrier rpcRetrier =
            new RemoteRetrier(
                remoteOptions,
                RemoteRetrier.RETRIABLE_GRPC_ERRORS,
                retryScheduler,
                Retrier.ALLOW_ALL_CALLS);
        executeRetrier = createExecuteRetrier(remoteOptions, retryScheduler);
        CallCredentials credentials = GoogleAuthUtils.newCallCredentials(authAndTlsOptions);
        ByteStreamUploader uploader =
            new ByteStreamUploader(
                remoteOptions.remoteInstanceName,
                channel.retain(),
                credentials,
                remoteOptions.remoteTimeout,
                rpcRetrier);
        cache =
            new GrpcRemoteCache(
                channel.retain(),
                credentials,
                remoteOptions,
                rpcRetrier,
                digestUtil,
                uploader.retain());
        Context requestContext =
            TracingMetadataUtils.contextWithMetadata(buildRequestId, invocationId, "bes-upload");
        buildEventArtifactUploaderFactoryDelegate.init(
            new ByteStreamBuildEventArtifactUploaderFactory(
                uploader, target, requestContext, remoteOptions.remoteInstanceName));
        uploader.release();
        channel.release();
      } else {
        executeRetrier = null;
        cache = null;
      }

      final GrpcRemoteExecutor executor;
      if (remoteOptions.remoteExecutor != null) {
        ManagedChannel channel =
            GoogleAuthUtils.newChannel(
                remoteOptions.remoteExecutor,
                authAndTlsOptions,
                interceptors.toArray(new ClientInterceptor[0]));
        RemoteRetrier retrier =
            new RemoteRetrier(
                remoteOptions,
                RemoteRetrier.RETRIABLE_GRPC_EXEC_ERRORS,
                retryScheduler,
                Retrier.ALLOW_ALL_CALLS);
        executor =
            new GrpcRemoteExecutor(
                channel,
                GoogleAuthUtils.newCallCredentials(authAndTlsOptions),
                retrier);
      } else {
        executor = null;
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

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    if (actionContextProvider != null) {
      builder.addActionContextProvider(actionContextProvider);
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
    public BuildEventArtifactUploader create(OptionsParsingResult options) {
      BuildEventArtifactUploaderFactory uploaderFactory0 = this.uploaderFactory;
      if (uploaderFactory0 == null) {
        return new LocalFilesArtifactUploader();
      }
      return uploaderFactory0.create(options);
    }
  }
}
