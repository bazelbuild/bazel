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
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystem.HashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import com.google.devtools.remoteexecution.v1test.Digest;
import java.io.IOException;

/** RemoteModule provides distributed cache and remote execution for Bazel. */
public final class RemoteModule extends BlazeModule {
  @VisibleForTesting
  static final class CasPathConverter implements PathConverter {
    // Not final; unfortunately, the Bazel startup process requires us to create this object before
    // we have the options available, so we have to create it first, and then set the options
    // afterwards. At the time of this writing, I believe that we aren't using the PathConverter
    // before the options are available, so this should be safe.
    // TODO(ulfjack): Change the Bazel startup process to make the options available when we create
    // the PathConverter.
    RemoteOptions options;

    @Override
    public String apply(Path path) {
      if (options == null || !remoteEnabled(options)) {
        return null;
      }
      String server = options.remoteCache;
      String remoteInstanceName = options.remoteInstanceName;
      try {
        Digest digest = Digests.computeDigest(path);
        return remoteInstanceName.isEmpty()
            ? String.format(
                "//%s/blobs/%s/%d",
                server,
                digest.getHash(),
                digest.getSizeBytes())
            : String.format(
                "//%s/%s/blobs/%s/%d",
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
  private CommandEnvironment env;

  @Override
  public void serverInit(OptionsProvider startupOptions, ServerBuilder builder)
      throws AbruptExitException {
    builder.addPathToUriConverter(converter);
  }

  @Override
  public void beforeCommand(Command command, CommandEnvironment env) {
    this.env = env;
    env.getEventBus().register(this);
  }

  @Override
  public void handleOptions(OptionsProvider optionsProvider) {
    converter.options = optionsProvider.getOptions(RemoteOptions.class);
  }

  @Override
  public void afterCommand() {
    this.env = null;
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    builder.addActionContextProvider(new RemoteActionContextProvider(env, request));
  }

  @Subscribe
  public void buildStarting(BuildStartingEvent event) {
    RemoteOptions options = event.getRequest().getOptions(RemoteOptions.class);

    if (remoteEnabled(options)) {
      HashFunction hf = FileSystem.getDigestFunction();
      if (hf != HashFunction.SHA1) {
        env.getBlazeModuleEnvironment().exit(new AbruptExitException(
            "Remote cache/execution requires SHA1 digests, got " + hf
            + ", run with --host_jvm_args=-Dbazel.DigestFunction=SHA1",
            ExitCode.COMMAND_LINE_ERROR));
      }
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
        || GrpcActionCache.isRemoteCacheOptions(options);
  }
}
