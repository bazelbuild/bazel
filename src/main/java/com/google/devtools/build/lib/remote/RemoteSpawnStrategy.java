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
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.remote.ContentDigests.ActionKey;
import com.google.devtools.build.lib.remote.RemoteProtocol.Action;
import com.google.devtools.build.lib.remote.RemoteProtocol.ActionResult;
import com.google.devtools.build.lib.remote.RemoteProtocol.Command;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecuteRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ExecutionStatus;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
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
  private final RemoteActionCache remoteActionCache;
  private final RemoteWorkExecutor remoteWorkExecutor;
  private final boolean verboseFailures;

  RemoteSpawnStrategy(
      Map<String, String> clientEnv,
      Path execRoot,
      RemoteOptions options,
      boolean verboseFailures,
      RemoteActionCache actionCache,
      RemoteWorkExecutor workExecutor,
      String productName) {
    this.execRoot = execRoot;
    this.standaloneStrategy = new StandaloneSpawnStrategy(execRoot, verboseFailures, productName);
    this.verboseFailures = verboseFailures;
    this.remoteActionCache = actionCache;
    this.remoteWorkExecutor = workExecutor;
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
    // TODO(olaola): Need to set platform as well!
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
      Spawn spawn, ActionExecutionContext actionExecutionContext, ActionKey actionKey)
      throws ExecException, InterruptedException {
    standaloneStrategy.exec(spawn, actionExecutionContext);
    if (remoteActionCache != null && actionKey != null) {
      ArrayList<Path> outputFiles = new ArrayList<>();
      for (ActionInput output : spawn.getOutputFiles()) {
        outputFiles.add(execRoot.getRelative(output.getExecPathString()));
      }
      try {
        ActionResult.Builder result = ActionResult.newBuilder();
        remoteActionCache.uploadAllResults(execRoot, outputFiles, result);
        remoteActionCache.setCachedActionResult(actionKey, result.build());
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

  private void passRemoteOutErr(ActionResult result, FileOutErr outErr) {
    if (remoteActionCache == null) {
      return;
    }
    try {
      ImmutableList<byte[]> streams =
          remoteActionCache.downloadBlobs(
              ImmutableList.of(result.getStdoutDigest(), result.getStderrDigest()));
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
    if (!spawn.isRemotable() || remoteActionCache == null) {
      standaloneStrategy.exec(spawn, actionExecutionContext);
      return;
    }

    ActionKey actionKey = null;
    String mnemonic = spawn.getMnemonic();
    Executor executor = actionExecutionContext.getExecutor();
    EventHandler eventHandler = executor.getEventHandler();
    executor.getEventBus().post(
        ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), "remote"));

    try {
      // Temporary hack: the TreeNodeRepository should be created and maintained upstream!
      TreeNodeRepository repository = new TreeNodeRepository(execRoot);
      List<ActionInput> inputs =
          ActionInputHelper.expandArtifacts(
              spawn.getInputFiles(), actionExecutionContext.getArtifactExpander());
      TreeNode inputRoot = repository.buildFromActionInputs(inputs);
      repository.computeMerkleDigests(inputRoot);
      Command command = buildCommand(spawn.getArguments(), spawn.getEnvironment());
      Action action =
          buildAction(
              spawn.getOutputFiles(),
              ContentDigests.computeDigest(command),
              repository.getMerkleDigest(inputRoot));

      // Look up action cache, and reuse the action output if it is found.
      actionKey = ContentDigests.computeActionKey(action);
      ActionResult result = remoteActionCache.getCachedActionResult(actionKey);
      boolean acceptCached = true;
      if (result != null) {
        // We don't cache failed actions, so we know the outputs exist.
        // For now, download all outputs locally; in the future, we can reuse the digests to
        // just update the TreeNodeRepository and continue the build.
        try {
          remoteActionCache.downloadAllResults(result, execRoot);
          return;
        } catch (CacheNotFoundException e) {
          acceptCached = false; // Retry the action remotely and invalidate the results.
        }
      }

      if (remoteWorkExecutor == null) {
        execLocally(spawn, actionExecutionContext, actionKey);
        return;
      }

      // Upload the command and all the inputs into the remote cache.
      remoteActionCache.uploadBlob(command.toByteArray());
      // TODO(olaola): this should use the ActionInputFileCache for SHA1 digests!
      remoteActionCache.uploadTree(repository, execRoot, inputRoot);
      // TODO(olaola): set BuildInfo and input total bytes as well.
      ExecuteRequest.Builder request =
          ExecuteRequest.newBuilder()
              .setAction(action)
              .setAcceptCached(acceptCached)
              .setTotalInputFileCount(inputs.size())
              .setTimeoutMillis(1000 * Spawns.getTimeoutSeconds(spawn, 120));
      // TODO(olaola): set sensible local and remote timouts.
      ExecuteReply reply = remoteWorkExecutor.executeRemotely(request.build());
      ExecutionStatus status = reply.getStatus();
      result = reply.getResult();
      // We do not want to pass on the remote stdout and strerr if we are going to retry the
      // action.
      if (status.getSucceeded()) {
        passRemoteOutErr(result, actionExecutionContext.getFileOutErr());
        remoteActionCache.downloadAllResults(result, execRoot);
        return;
      }
      if (status.getError() == ExecutionStatus.ErrorCode.EXEC_FAILED) {
        passRemoteOutErr(result, actionExecutionContext.getFileOutErr());
        throw new UserExecException(status.getErrorDetail());
      }
      // For now, we retry locally on all other remote errors.
      // TODO(olaola): add remote retries on cache miss errors.
      execLocally(spawn, actionExecutionContext, actionKey);
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
      execLocally(spawn, actionExecutionContext, actionKey);
    } catch (CacheNotFoundException e) {
      eventHandler.handle(Event.warn(mnemonic + " remote work results cache miss (" + e + ")"));
      execLocally(spawn, actionExecutionContext, actionKey);
    } catch (UnsupportedOperationException e) {
      eventHandler.handle(
          Event.warn(mnemonic + " unsupported operation for action cache (" + e + ")"));
    }
  }

  @Override
  public boolean willExecuteRemotely(boolean remotable) {
    // Returning true here just helps to estimate the cost of this computation is zero.
    return remotable;
  }

  @Override
  public boolean shouldPropagateExecException() {
    return false;
  }
}
