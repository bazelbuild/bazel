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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
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
import com.google.devtools.remoteexecution.v1test.Digest;
import io.grpc.Channel;
import io.grpc.ClientInterceptors;
import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.util.function.Predicate;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

/** RemoteModule provides distributed cache and remote execution for Bazel. */
public final class RemoteModule extends BlazeModule {
  private static final Logger logger = Logger.getLogger(RemoteModule.class.getName());
  private AsynchronousFileOutputStream rpcLogFile;

  @VisibleForTesting
  static final class CasPathConverter implements PathConverter {
    // Not final; unfortunately, the Bazel startup process requires us to create this object before
    // we have the options available, so we have to create it first, and then set the options
    // afterwards. At the time of this writing, I believe that we aren't using the PathConverter
    // before the options are available, so this should be safe.
    // TODO(ulfjack): Change the Bazel startup process to make the options available when we create
    // the PathConverter.
    RemoteOptions options;
    DigestUtil digestUtil;

    @Override
    public String apply(Path path) {
      if (options == null || digestUtil == null || !remoteEnabled(options)) {
        return null;
      }
      String server = options.remoteCache;
      String remoteInstanceName = options.remoteInstanceName;
      try {
        Digest digest = digestUtil.compute(path);
        return remoteInstanceName.isEmpty()
            ? String.format(
                "bytestream://%s/blobs/%s/%d",
                server,
                digest.getHash(),
                digest.getSizeBytes())
            : String.format(
                "bytestream://%s/%s/blobs/%s/%d",
                server,
                remoteInstanceName,
                digest.getHash(),
                digest.getSizeBytes());
      } catch (IOException e) {
        // TODO(ulfjack): Don't fail silently!
        return null;
      }
    }
  }

  private final CasPathConverter converter = new CasPathConverter();
  private final ListeningScheduledExecutorService retryScheduler =
      MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  private RemoteActionContextProvider actionContextProvider;

  @Override
  public void serverInit(OptionsProvider startupOptions, ServerBuilder builder)
      throws AbruptExitException {
    builder.addPathToUriConverter(converter);
  }

  private static final Predicate<? super Exception> RETRIABLE_EXEC_ERRORS =
      e -> {
        if (e instanceof CacheNotFoundException) {
          return true;
        }
        if (e instanceof StatusRuntimeException) {
          return Status.fromThrowable(e).getCode() == Status.Code.FAILED_PRECONDITION;
        }
        return false;
      };

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
    RemoteOptions remoteOptions = env.getOptions().getOptions(RemoteOptions.class);
    AuthAndTLSOptions authAndTlsOptions = env.getOptions().getOptions(AuthAndTLSOptions.class);
    HashFunction hashFn = env.getRuntime().getFileSystem().getDigestFunction();
    DigestUtil digestUtil = new DigestUtil(hashFn);
    converter.options = remoteOptions;
    converter.digestUtil = digestUtil;

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
      LoggingInterceptor logger = null;
      if (!remoteOptions.experimentalRemoteGrpcLog.isEmpty()) {
        rpcLogFile = new AsynchronousFileOutputStream(remoteOptions.experimentalRemoteGrpcLog);
        logger = new LoggingInterceptor(rpcLogFile, env.getRuntime().getClock());
      }

      final Retrier executeRetrier;
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
        Channel ch = GoogleAuthUtils.newChannel(target, authAndTlsOptions);
        if (logger != null) {
          ch = ClientInterceptors.intercept(ch, logger);
        }
        RemoteRetrier retrier =
            new RemoteRetrier(
                remoteOptions,
                RemoteRetrier.RETRIABLE_GRPC_ERRORS,
                retryScheduler,
                Retrier.ALLOW_ALL_CALLS);
        executeRetrier =
            new RemoteRetrier(
                remoteOptions,
                RemoteModule.RETRIABLE_EXEC_ERRORS,
                retryScheduler,
                Retrier.ALLOW_ALL_CALLS);
        cache =
            new GrpcRemoteCache(
                ch,
                GoogleAuthUtils.newCallCredentials(authAndTlsOptions),
                remoteOptions,
                retrier,
                digestUtil);
      } else {
        cache = null;
      }

      final GrpcRemoteExecutor executor;
      if (remoteOptions.remoteExecutor != null) {
        Channel ch = GoogleAuthUtils.newChannel(remoteOptions.remoteExecutor, authAndTlsOptions);
        RemoteRetrier retrier =
            new RemoteRetrier(
                remoteOptions,
                RemoteRetrier.RETRIABLE_GRPC_ERRORS,
                retryScheduler,
                Retrier.ALLOW_ALL_CALLS);
        if (logger != null) {
          ch = ClientInterceptors.intercept(ch, logger);
        }
        executor =
            new GrpcRemoteExecutor(
                ch,
                GoogleAuthUtils.newCallCredentials(authAndTlsOptions),
                remoteOptions.remoteTimeout,
                retrier);
      } else {
        executor = null;
      }
      actionContextProvider =
          new RemoteActionContextProvider(env, cache, executor, executeRetrier, digestUtil, logDir);
    } catch (IOException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      env.getBlazeModuleEnvironment().exit(new AbruptExitException(
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

  public static boolean remoteEnabled(RemoteOptions options) {
    return SimpleBlobStoreFactory.isRemoteCacheOptions(options)
        || GrpcRemoteCache.isRemoteCacheOptions(options);
  }
}
