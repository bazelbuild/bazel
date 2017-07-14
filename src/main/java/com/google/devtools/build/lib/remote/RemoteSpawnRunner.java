// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
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
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeSet;
import javax.annotation.Nullable;

/** A client for the remote execution service. */
@ThreadSafe
final class RemoteSpawnRunner implements SpawnRunner {
  private final Path execRoot;
  private final RemoteOptions options;
  // TODO(olaola): This will be set on a per-action basis instead.
  private final Platform platform;
  private final SpawnRunner fallbackRunner;

  @Nullable private final RemoteActionCache remoteCache;
  @Nullable private final GrpcRemoteExecutor remoteExecutor;

  RemoteSpawnRunner(
      Path execRoot,
      RemoteOptions options,
      SpawnRunner fallbackRunner,
      @Nullable RemoteActionCache remoteCache,
      @Nullable GrpcRemoteExecutor remoteExecutor) {
    this.execRoot = execRoot;
    this.options = options;
    this.platform = options.parseRemotePlatformOverride();
    this.fallbackRunner = fallbackRunner;
    this.remoteCache = remoteCache;
    this.remoteExecutor = remoteExecutor;
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionPolicy policy)
      throws ExecException, InterruptedException, IOException {
    if (!spawn.isRemotable() || remoteCache == null) {
      return fallbackRunner.exec(spawn, policy);
    }

    policy.report(ProgressStatus.EXECUTING);
    // Temporary hack: the TreeNodeRepository should be created and maintained upstream!
    ActionInputFileCache inputFileCache = policy.getActionInputFileCache();
    TreeNodeRepository repository = new TreeNodeRepository(execRoot, inputFileCache);
    SortedMap<PathFragment, ActionInput> inputMap = policy.getInputMapping();
    TreeNode inputRoot = repository.buildFromActionInputs(inputMap);
    repository.computeMerkleDigests(inputRoot);
    Command command = buildCommand(spawn.getArguments(), spawn.getEnvironment());
    Action action =
        buildAction(
            spawn.getOutputFiles(),
            Digests.computeDigest(command),
            repository.getMerkleDigest(inputRoot),
            Spawns.getTimeoutSeconds(spawn));

    // Look up action cache, and reuse the action output if it is found.
    ActionKey actionKey = Digests.computeActionKey(action);
    try {
      boolean acceptCachedResult = options.remoteAcceptCached && Spawns.mayBeCached(spawn);
      ActionResult result =
          acceptCachedResult
              ? remoteCache.getCachedActionResult(actionKey)
              : null;
      if (result != null) {
        // We don't cache failed actions, so we know the outputs exist.
        // For now, download all outputs locally; in the future, we can reuse the digests to
        // just update the TreeNodeRepository and continue the build.
        try {
          remoteCache.download(result, execRoot, policy.getFileOutErr());
          return new SpawnResult.Builder()
              .setStatus(Status.SUCCESS)  // Even if the action failed with non-zero exit code.
              .setExitCode(result.getExitCode())
              .build();
        } catch (CacheNotFoundException e) {
          acceptCachedResult = false; // Retry the action remotely and invalidate the results.
        }
      }

      if (remoteExecutor == null) {
        return execLocally(spawn, policy, remoteCache, actionKey);
      }

      // Upload the command and all the inputs into the remote cache.
      remoteCache.ensureInputsPresent(repository, execRoot, inputRoot, command);
      // TODO(olaola): set BuildInfo and input total bytes as well.
      ExecuteRequest.Builder request =
          ExecuteRequest.newBuilder()
              .setInstanceName(options.remoteInstanceName)
              .setAction(action)
              .setTotalInputFileCount(inputMap.size())
              .setSkipCacheLookup(!acceptCachedResult);
      ExecuteResponse reply = remoteExecutor.executeRemotely(request.build());
      result = reply.getResult();
      if (options.remoteLocalFallback && result.getExitCode() != 0) {
        return execLocally(spawn, policy, remoteCache, actionKey);
      }
      remoteCache.download(result, execRoot, policy.getFileOutErr());
      return new SpawnResult.Builder()
          .setStatus(Status.SUCCESS)  // Even if the action failed with non-zero exit code.
          .setExitCode(result.getExitCode())
          .build();
    } catch (IOException e) {
      if (options.remoteLocalFallback) {
        return execLocally(spawn, policy, remoteCache, actionKey);
      }
      throw e;
    } catch (CacheNotFoundException e) {
      if (options.remoteLocalFallback) {
        return execLocally(spawn, policy, remoteCache, actionKey);
      }
      throw new IOException(e);
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
    ArrayList<String> outputPaths = new ArrayList<>();
    for (ActionInput output : outputs) {
      outputPaths.add(output.getExecPathString());
    }
    Collections.sort(outputPaths);
    // TODO: output directories should be handled here, when they are supported.
    action.addAllOutputFiles(outputPaths);
    if (platform != null) {
      action.setPlatform(platform);
    }
    if (timeoutSeconds > 0) {
      action.setTimeout(Duration.newBuilder().setSeconds(timeoutSeconds));
    }
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
  private SpawnResult execLocally(
      Spawn spawn,
      SpawnExecutionPolicy policy,
      RemoteActionCache remoteCache,
      ActionKey actionKey)
      throws ExecException, IOException, InterruptedException {
    SpawnResult result = fallbackRunner.exec(spawn, policy);
    if (options.remoteUploadLocalResults && Spawns.mayBeCached(spawn) && remoteCache != null
        && actionKey != null) {
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
      remoteCache.upload(actionKey, execRoot, outputFiles, policy.getFileOutErr());
    }
    return result;
  }

  /** Release resources associated with this spawn runner. */
  public void close() {
    if (remoteCache != null) {
      remoteCache.close();
    }
  }
}
