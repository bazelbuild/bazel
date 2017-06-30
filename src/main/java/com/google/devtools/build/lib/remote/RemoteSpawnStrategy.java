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

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.remoteexecution.v1test.Action;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Command;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.ExecuteResponse;
import com.google.devtools.remoteexecution.v1test.Platform;
import com.google.protobuf.Duration;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeSet;

/**
 * Strategy that uses a distributed cache for sharing action input and output files. Optionally this
 * strategy also support offloading the work to a remote worker.
 */
@ExecutionStrategy(
  name = {"remote"},
  contextType = SpawnActionContext.class
)
final class RemoteSpawnStrategy implements SpawnActionContext {
  private final Path execRoot;
  private final SpawnActionContext fallbackStrategy;
  private final boolean verboseFailures;
  private final RemoteOptions remoteOptions;
  // TODO(olaola): This will be set on a per-action basis instead.
  private final Platform platform;
  private final ChannelOptions channelOptions;
  private final SpawnInputExpander spawnInputExpander = new SpawnInputExpander(/*strict=*/ false);

  private final RemoteActionCache remoteCache;
  private final GrpcRemoteExecutor workExecutor;

  RemoteSpawnStrategy(
      Path execRoot,
      RemoteOptions remoteOptions,
      AuthAndTLSOptions authTlsOptions,
      boolean verboseFailures,
      SpawnActionContext fallbackStrategy) {
    this.execRoot = execRoot;
    this.fallbackStrategy = fallbackStrategy;
    this.verboseFailures = verboseFailures;
    this.remoteOptions = remoteOptions;
    channelOptions = ChannelOptions.create(authTlsOptions);
    if (remoteOptions.experimentalRemotePlatformOverride != null) {
      Platform.Builder platformBuilder = Platform.newBuilder();
      try {
        TextFormat.getParser()
            .merge(remoteOptions.experimentalRemotePlatformOverride, platformBuilder);
      } catch (ParseException e) {
        throw new IllegalArgumentException(
            "Failed to parse --experimental_remote_platform_override", e);
      }
      platform = platformBuilder.build();
    } else {
      platform = null;
    }
    // Initialize remote cache and execution handlers. We use separate handlers for every
    // action to enable server-side parallelism (need a different gRPC channel per action).
    if (SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)) {
      remoteCache = new SimpleBlobStoreActionCache(SimpleBlobStoreFactory.create(remoteOptions));
    } else if (GrpcActionCache.isRemoteCacheOptions(remoteOptions)) {
      remoteCache =
          new GrpcActionCache(
              GrpcUtils.createChannel(remoteOptions.remoteCache, channelOptions),
              channelOptions,
              remoteOptions);
    } else {
      remoteCache = null;
    }
    // Otherwise remoteCache remains null and remote caching/execution are disabled.

    if (remoteCache != null && GrpcRemoteExecutor.isRemoteExecutionOptions(remoteOptions)) {
      workExecutor =
          new GrpcRemoteExecutor(
              GrpcUtils.createChannel(remoteOptions.remoteExecutor, channelOptions),
              channelOptions,
              remoteOptions);
    } else {
      workExecutor = null;
    }
  }

  /** Release resources associated with this spawn strategy. */
  public void close() {
    if (remoteCache != null) {
      remoteCache.close();
    }
  }

  private Action buildAction(
      Collection<? extends ActionInput> outputs,
      Digest command,
      Digest inputRoot,
      long timeoutSeconds) {
    Action.Builder action = Action.newBuilder();
    action.setCommandDigest(command);
    action.setInputRootDigest(inputRoot);
    // Somewhat ugly: we rely on the stable order of outputs here for remote action caching.
    for (ActionInput output : outputs) {
      // TODO: output directories should be handled here, when they are supported.
      action.addOutputFiles(output.getExecPathString());
    }
    if (platform != null) {
      action.setPlatform(platform);
    }
    action.setTimeout(Duration.newBuilder().setSeconds(timeoutSeconds));
    return action.build();
  }

  private Command buildCommand(List<String> arguments, ImmutableMap<String, String> environment) {
    Command.Builder command = Command.newBuilder();
    command.addAllArguments(arguments);
    // Sorting the environment pairs by variable name.
    TreeSet<String> variables = new TreeSet<>(environment.keySet());
    for (String var : variables) {
      command.addEnvironmentVariablesBuilder().setName(var).setValue(environment.get(var));
    }
    return command.build();
  }

  /**
   * Fallback: execute the spawn locally. If an ActionKey is provided, try to upload results to
   * remote action cache.
   */
  private void execLocally(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      RemoteActionCache remoteCache,
      ActionKey actionKey)
      throws ExecException, InterruptedException {
    fallbackStrategy.exec(spawn, actionExecutionContext);
    if (remoteOptions.remoteUploadLocalResults && remoteCache != null && actionKey != null) {
      ArrayList<Path> outputFiles = new ArrayList<>();
      for (ActionInput output : spawn.getOutputFiles()) {
        Path outputFile = execRoot.getRelative(output.getExecPathString());
        // Ignore non-existent files.
        // TODO(ulfjack): This is not ideal - in general, all spawn strategies should stat the
        // output files and return a list of existing files. We shouldn't re-stat the files here.
        if (!outputFile.exists()) {
          continue;
        }
        outputFiles.add(outputFile);
      }
      try {
        remoteCache.upload(
            actionKey, execRoot, outputFiles, actionExecutionContext.getFileOutErr());
      } catch (IOException e) {
        throw new UserExecException("Unexpected IO error.", e);
      } catch (UnsupportedOperationException e) {
        actionExecutionContext
            .getEventHandler()
            .handle(
                Event.warn(
                    spawn.getMnemonic() + " unsupported operation for action cache (" + e + ")"));
      } catch (StatusRuntimeException e) {
        actionExecutionContext
            .getEventHandler()
            .handle(Event.warn(spawn.getMnemonic() + " failed uploading results (" + e + ")"));
      }
    }
  }

  @Override
  public String toString() {
    return "remote";
  }

  /** Executes the given {@code spawn}. */
  @Override
  public void exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    ActionKey actionKey = null;
    String mnemonic = spawn.getMnemonic();
    EventHandler eventHandler = actionExecutionContext.getEventHandler();

    if (!spawn.isRemotable() || remoteCache == null) {
      fallbackStrategy.exec(spawn, actionExecutionContext);
      return;
    }
    if (actionExecutionContext.reportsSubcommands()) {
      actionExecutionContext.reportSubcommand(spawn);
    }
    actionExecutionContext
        .getEventBus()
        .post(ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), "remote"));

    try {
      // Temporary hack: the TreeNodeRepository should be created and maintained upstream!
      ActionInputFileCache inputFileCache = actionExecutionContext.getActionInputFileCache();
      TreeNodeRepository repository = new TreeNodeRepository(execRoot, inputFileCache);
      SortedMap<PathFragment, ActionInput> inputMap =
          spawnInputExpander.getInputMapping(
              spawn,
              actionExecutionContext.getArtifactExpander(),
              actionExecutionContext.getActionInputFileCache(),
              actionExecutionContext.getContext(FilesetActionContext.class));
      TreeNode inputRoot = repository.buildFromActionInputs(inputMap);
      repository.computeMerkleDigests(inputRoot);
      Command command = buildCommand(spawn.getArguments(), spawn.getEnvironment());
      Action action =
          buildAction(
              spawn.getOutputFiles(),
              Digests.computeDigest(command),
              repository.getMerkleDigest(inputRoot),
              // TODO(olaola): set sensible local and remote timouts.
              Spawns.getTimeoutSeconds(spawn, 120));

      // Look up action cache, and reuse the action output if it is found.
      actionKey = Digests.computeActionKey(action);
      ActionResult result =
          this.remoteOptions.remoteAcceptCached
              ? remoteCache.getCachedActionResult(actionKey)
              : null;
      boolean acceptCachedResult = this.remoteOptions.remoteAcceptCached;
      if (result != null) {
        // We don't cache failed actions, so we know the outputs exist.
        // For now, download all outputs locally; in the future, we can reuse the digests to
        // just update the TreeNodeRepository and continue the build.
        try {
          remoteCache.download(result, execRoot, actionExecutionContext.getFileOutErr());
          return;
        } catch (CacheNotFoundException e) {
          acceptCachedResult = false; // Retry the action remotely and invalidate the results.
        }
      }

      if (workExecutor == null) {
        execLocally(spawn, actionExecutionContext, remoteCache, actionKey);
        return;
      }

      // Upload the command and all the inputs into the remote cache.
      remoteCache.ensureInputsPresent(repository, execRoot, inputRoot, command);
      // TODO(olaola): set BuildInfo and input total bytes as well.
      ExecuteRequest.Builder request =
          ExecuteRequest.newBuilder()
              .setInstanceName(remoteOptions.remoteInstanceName)
              .setAction(action)
              .setTotalInputFileCount(inputMap.size())
              .setSkipCacheLookup(!acceptCachedResult);
      ExecuteResponse reply = workExecutor.executeRemotely(request.build());
      result = reply.getResult();
      if (remoteOptions.remoteLocalFallback && result.getExitCode() != 0) {
        execLocally(spawn, actionExecutionContext, remoteCache, actionKey);
        return;
      }
      remoteCache.download(result, execRoot, actionExecutionContext.getFileOutErr());
      if (result.getExitCode() != 0) {
        String cwd = actionExecutionContext.getExecRoot().getPathString();
        String message =
            CommandFailureUtils.describeCommandFailure(
                verboseFailures, spawn.getArguments(), spawn.getEnvironment(), cwd);
        throw new UserExecException(message + ": Exit " + result.getExitCode());
      }
    } catch (RetryException e) {
      String stackTrace = "";
      if (verboseFailures) {
        stackTrace = "\n" + Throwables.getStackTraceAsString(e);
      }
      eventHandler.handle(Event.warn(mnemonic + " remote work failed (" + e + ")" + stackTrace));
      if (remoteOptions.remoteLocalFallback) {
        execLocally(spawn, actionExecutionContext, remoteCache, actionKey);
      } else {
        throw new UserExecException(e.getCause());
      }
    } catch (CacheNotFoundException e) {
      // TODO(olaola): handle this exception by reuploading / reexecuting the action remotely.
      eventHandler.handle(Event.warn(mnemonic + " remote work results cache miss (" + e + ")"));
      if (remoteOptions.remoteLocalFallback) {
        execLocally(spawn, actionExecutionContext, remoteCache, actionKey);
      } else {
        throw new UserExecException(e);
      }
    } catch (IOException e) {
      throw new UserExecException("Unexpected IO error.", e);
    } catch (UnsupportedOperationException e) {
      eventHandler.handle(
          Event.warn(mnemonic + " unsupported operation for action cache (" + e + ")"));
    }
  }

  @Override
  public boolean shouldPropagateExecException() {
    return false;
  }
}
