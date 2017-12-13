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
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GrpcUtils;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.FileSystem.HashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import com.google.devtools.remoteexecution.v1test.Digest;
import io.grpc.CallCredentials;
import io.grpc.Channel;
import java.io.IOException;
import java.util.logging.Logger;

/** RemoteModule provides distributed cache and remote execution for Bazel. */
public final class RemoteModule extends BlazeModule {
  private static final Logger logger = Logger.getLogger(RemoteModule.class.getName());

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

  private RemoteActionContextProvider actionContextProvider;

  @Override
  public void serverInit(OptionsProvider startupOptions, ServerBuilder builder)
      throws AbruptExitException {
    builder.addPathToUriConverter(converter);
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    env.getEventBus().register(this);
    String buildRequestId = env.getBuildRequestId().toString();
    String commandId = env.getCommandId().toString();
    logger.info("Command: buildRequestId = " + buildRequestId + ", commandId = " + commandId);
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


    try {
      boolean remoteOrLocalCache = SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions);
      boolean grpcCache = GrpcRemoteCache.isRemoteCacheOptions(remoteOptions);

      RemoteRetrier retrier =
          new RemoteRetrier(
              remoteOptions, RemoteRetrier.RETRIABLE_GRPC_ERRORS, Retrier.ALLOW_ALL_CALLS);
      CallCredentials creds = GrpcUtils.newCallCredentials(authAndTlsOptions);
      // TODO(davido): The naming is wrong here. "Remote"-prefix in RemoteActionCache class has no
      // meaning.
      final RemoteActionCache cache;
      if (remoteOrLocalCache) {
        cache =
            new SimpleBlobStoreActionCache(
                SimpleBlobStoreFactory.create(remoteOptions, env.getWorkingDirectory()),
                digestUtil);
      } else if (grpcCache || remoteOptions.remoteExecutor != null) {
        // If a remote executor but no remote cache is specified, assume both at the same target.
        String target = grpcCache ? remoteOptions.remoteCache : remoteOptions.remoteExecutor;
        Channel ch = GrpcUtils.newChannel(target, authAndTlsOptions);
        cache = new GrpcRemoteCache(ch, creds, remoteOptions, retrier, digestUtil);
      } else {
        cache = null;
      }

      final GrpcRemoteExecutor executor;
      if (remoteOptions.remoteExecutor != null) {
        executor = new GrpcRemoteExecutor(
            GrpcUtils.newChannel(remoteOptions.remoteExecutor, authAndTlsOptions),
            creds,
            remoteOptions.remoteTimeout,
            retrier);
      } else {
        executor = null;
      }

      actionContextProvider = new RemoteActionContextProvider(env, cache, executor, digestUtil);
    } catch (IOException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      env.getBlazeModuleEnvironment().exit(new AbruptExitException(ExitCode.COMMAND_LINE_ERROR));
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
