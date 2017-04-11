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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.remote.ContentDigests.ActionKey;
import com.google.devtools.build.lib.remote.RemoteProtocol.Action;
import com.google.devtools.build.lib.remote.RemoteProtocol.ActionResult;
import com.google.devtools.build.lib.remote.RemoteProtocol.Command;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionStatus;
import com.google.devtools.build.lib.remote.RemoteProtocol.Platform;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
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
  private final StandaloneSpawnStrategy standaloneStrategy;
  private final boolean verboseFailures;
  private final RemoteOptions options;
  // TODO(olaola): This will be set on a per-action basis instead.
  private final Platform platform;
  private final SpawnInputExpander spawnInputExpander = new SpawnInputExpander(/*strict=*/false);

  RemoteSpawnStrategy(
      Map<String, String> clientEnv,
      Path execRoot,
      RemoteOptions options,
      boolean verboseFailures,
      String productName) {
    this.execRoot = execRoot;
    this.standaloneStrategy = new StandaloneSpawnStrategy(execRoot, verboseFailures, productName);
    this.verboseFailures = verboseFailures;
    this.options = options;
    if (options.experimentalRemotePlatformOverride != null) {
      Platform.Builder platformBuilder = Platform.newBuilder();
      try {
        TextFormat.getParser().merge(options.experimentalRemotePlatformOverride, platformBuilder);
      } catch (ParseException e) {
        throw new RuntimeException("Failed to parse --experimental_remote_platform_override", e);
      }
      platform = platformBuilder.build();
    } else {
      platform = null;
    }
  }

  private Action buildAction(
      Collection<? extends ActionInput> outputs, ContentDigest command, ContentDigest inputRoot) {
    Action.Builder action = Action.newBuilder();
    action.setCommandDigest(command);
    action.setInputRootDigest(inputRoot);
    // Somewhat ugly: we rely on the stable order of outputs here for remote action caching.
    for (ActionInput output : outputs) {
      action.addOutputPath(output.getExecPathString());
    }
    if (platform != null) {
      action.setPlatform(platform);
    }
    return action.build();
  }

  private Command buildCommand(List<String> arguments, ImmutableMap<String, String> environment) {
    Command.Builder command = Command.newBuilder();
    command.addAllArgv(arguments);
    // Sorting the environment pairs by variable name.
    TreeSet<String> variables = new TreeSet<>(environment.keySet());
    for (String var : variables) {
      command.addEnvironmentBuilder().setVariable(var).setValue(environment.get(var));
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
      RemoteActionCache actionCache,
      ActionKey actionKey)
      throws ExecException, InterruptedException {
    standaloneStrategy.exec(spawn, actionExecutionContext);
    if (options.remoteLocalExecUploadResults && actionCache != null && actionKey != null) {
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
        ActionResult.Builder result = ActionResult.newBuilder();
        actionCache.uploadAllResults(execRoot, outputFiles, result);
        actionCache.setCachedActionResult(actionKey, result.build());
        // Handle all cache errors here.
      } catch (IOException e) {
        throw new UserExecException("Unexpected IO error.", e);
      } catch (UnsupportedOperationException e) {
        actionExecutionContext
            .getExecutor()
            .getEventHandler()
            .handle(
                Event.warn(
                    spawn.getMnemonic() + " unsupported operation for action cache (" + e + ")"));
      } catch (StatusRuntimeException e) {
        actionExecutionContext
            .getExecutor()
            .getEventHandler()
            .handle(Event.warn(spawn.getMnemonic() + " failed uploading results (" + e + ")"));
      }
    }
  }

  private static void passRemoteOutErr(
      RemoteActionCache cache, ActionResult result, FileOutErr outErr) {
    try {
      ImmutableList<byte[]> streams =
          cache.downloadBlobs(ImmutableList.of(result.getStdoutDigest(), result.getStderrDigest()));
      outErr.printOut(new String(streams.get(0), UTF_8));
      outErr.printErr(new String(streams.get(1), UTF_8));
    } catch (CacheNotFoundException e) {
      // Ignoring.
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
    Executor executor = actionExecutionContext.getExecutor();
    EventHandler eventHandler = executor.getEventHandler();

    RemoteActionCache actionCache = null;
    GrpcRemoteExecutor workExecutor = null;
    if (spawn.isRemotable()) {
      // Initialize remote cache and execution handlers. We use separate handlers for every
      // action to enable server-side parallelism (need a different gRPC channel per action).
      try {
        if (SimpleBlobStoreFactory.isRemoteCacheOptions(options)) {
          actionCache = new SimpleBlobStoreActionCache(SimpleBlobStoreFactory.create(options));
        } else if (GrpcActionCache.isRemoteCacheOptions(options)) {
          actionCache = new GrpcActionCache(options);
        }
        // Otherwise actionCache remains null and remote caching/execution are disabled.

        if (actionCache != null && GrpcRemoteExecutor.isRemoteExecutionOptions(options)) {
          workExecutor = new GrpcRemoteExecutor(
              RemoteUtils.createChannelLegacy(options.remoteWorker), options);
        }
      } catch (InvalidConfigurationException e) {
        eventHandler.handle(Event.warn(e.toString()));
      }
    }
    if (!spawn.isRemotable() || actionCache == null) {
      standaloneStrategy.exec(spawn, actionExecutionContext);
      return;
    }
    if (executor.reportsSubcommands()) {
      executor.reportSubcommand(spawn);
    }
    executor.getEventBus().post(
        ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), "remote"));

    try {
      // Temporary hack: the TreeNodeRepository should be created and maintained upstream!
      ActionInputFileCache inputFileCache = actionExecutionContext.getActionInputFileCache();
      TreeNodeRepository repository = new TreeNodeRepository(execRoot, inputFileCache);
      SortedMap<PathFragment, ActionInput> inputMap =
          spawnInputExpander.getInputMapping(
              spawn,
              actionExecutionContext.getArtifactExpander(),
              actionExecutionContext.getActionInputFileCache(),
              actionExecutionContext.getExecutor().getContext(FilesetActionContext.class));
      TreeNode inputRoot = repository.buildFromActionInputs(inputMap);
      repository.computeMerkleDigests(inputRoot);
      Command command = buildCommand(spawn.getArguments(), spawn.getEnvironment());
      Action action =
          buildAction(
              spawn.getOutputFiles(),
              ContentDigests.computeDigest(command),
              repository.getMerkleDigest(inputRoot));

      // Look up action cache, and reuse the action output if it is found.
      actionKey = ContentDigests.computeActionKey(action);
      ActionResult result =
          this.options.remoteAcceptCached ? actionCache.getCachedActionResult(actionKey) : null;
      boolean acceptCachedResult = this.options.remoteAcceptCached;
      if (result != null) {
        // We don't cache failed actions, so we know the outputs exist.
        // For now, download all outputs locally; in the future, we can reuse the digests to
        // just update the TreeNodeRepository and continue the build.
        try {
          actionCache.downloadAllResults(result, execRoot);
          return;
        } catch (CacheNotFoundException e) {
          acceptCachedResult = false; // Retry the action remotely and invalidate the results.
        }
      }

      if (workExecutor == null) {
        execLocally(spawn, actionExecutionContext, actionCache, actionKey);
        return;
      }

      // Upload the command and all the inputs into the remote cache.
      actionCache.uploadBlob(command.toByteArray());
      // TODO(olaola): this should use the ActionInputFileCache for SHA1 digests!
      actionCache.uploadTree(repository, execRoot, inputRoot);
      // TODO(olaola): set BuildInfo and input total bytes as well.
      ExecuteRequest.Builder request =
          ExecuteRequest.newBuilder()
              .setAction(action)
              .setAcceptCached(acceptCachedResult)
              .setTotalInputFileCount(inputMap.size())
              .setTimeoutMillis(1000 * Spawns.getTimeoutSeconds(spawn, 120));
      // TODO(olaola): set sensible local and remote timouts.
      ExecuteReply reply = workExecutor.executeRemotely(request.build());
      ExecutionStatus status = reply.getStatus();
      result = reply.getResult();
      // We do not want to pass on the remote stdout and strerr if we are going to retry the
      // action.
      if (status.getSucceeded()) {
        passRemoteOutErr(actionCache, result, actionExecutionContext.getFileOutErr());
        actionCache.downloadAllResults(result, execRoot);
        return;
      }
      if (status.getError() == ExecutionStatus.ErrorCode.EXEC_FAILED
          || !options.remoteAllowLocalFallback) {
        passRemoteOutErr(actionCache, result, actionExecutionContext.getFileOutErr());
        throw new UserExecException(status.getErrorDetail());
      }
      // For now, we retry locally on all other remote errors.
      // TODO(olaola): add remote retries on cache miss errors.
      execLocally(spawn, actionExecutionContext, actionCache, actionKey);
    } catch (IOException e) {
      throw new UserExecException("Unexpected IO error.", e);
    } catch (InterruptedException e) {
      eventHandler.handle(Event.warn(mnemonic + " remote work interrupted (" + e + ")"));
      Thread.currentThread().interrupt();
      throw e;
    } catch (StatusRuntimeException e) {
      String stackTrace = "";
      if (verboseFailures) {
        stackTrace = "\n" + Throwables.getStackTraceAsString(e);
      }
      eventHandler.handle(Event.warn(mnemonic + " remote work failed (" + e + ")" + stackTrace));
      if (options.remoteAllowLocalFallback) {
        execLocally(spawn, actionExecutionContext, actionCache, actionKey);
      } else {
        throw new UserExecException(e);
      }
    } catch (CacheNotFoundException e) {
      eventHandler.handle(Event.warn(mnemonic + " remote work results cache miss (" + e + ")"));
      if (options.remoteAllowLocalFallback) {
        execLocally(spawn, actionExecutionContext, actionCache, actionKey);
      } else {
        throw new UserExecException(e);
      }
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
