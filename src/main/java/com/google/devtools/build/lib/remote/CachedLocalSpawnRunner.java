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
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.remoteexecution.v1test.Action;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Command;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.Platform;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeSet;

/**
 * A {@link SpawnRunner} implementation that adds a remote cache on top of an underlying local
 * {@link SpawnRunner} implementation.
 */
@ThreadSafe // If RemoteActionCache and SpawnRunner implementations are thread-safe.
final class CachedLocalSpawnRunner implements SpawnRunner {
  private final Path execRoot;
  private final RemoteOptions options;
  // TODO(olaola): This will be set on a per-action basis instead.
  private final Platform platform;

  private final RemoteActionCache remoteCache;
  private final SpawnRunner delegate;

  CachedLocalSpawnRunner(
      Path execRoot, RemoteOptions options, RemoteActionCache remoteCache, SpawnRunner delegate) {
    this.execRoot = execRoot;
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
    this.remoteCache = remoteCache;
    this.delegate = delegate;
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionPolicy policy)
      throws InterruptedException, IOException, ExecException {
    ActionKey actionKey = null;
    String mnemonic = spawn.getMnemonic();

    try {
      // Temporary hack: the TreeNodeRepository should be created and maintained upstream!
      TreeNodeRepository repository =
          new TreeNodeRepository(execRoot, policy.getActionInputFileCache());
      SortedMap<PathFragment, ActionInput> inputMap = policy.getInputMapping();
      TreeNode inputRoot = repository.buildFromActionInputs(inputMap);
      repository.computeMerkleDigests(inputRoot);
      Command command = buildCommand(spawn.getArguments(), spawn.getEnvironment());
      Action action =
          buildAction(
              spawn.getOutputFiles(),
              Digests.computeDigest(command),
              repository.getMerkleDigest(inputRoot),
              policy.getTimeout());

      // Look up action cache, and reuse the action output if it is found.
      actionKey = Digests.computeActionKey(action);
      ActionResult result =
          this.options.remoteAcceptCached ? remoteCache.getCachedActionResult(actionKey) : null;
      if (result != null) {
        // We don't cache failed actions, so we know the outputs exist.
        // For now, download all outputs locally; in the future, we can reuse the digests to
        // just update the TreeNodeRepository and continue the build.
        try {
          remoteCache.download(result, execRoot, policy.getFileOutErr());
          return new SpawnResult.Builder()
              .setStatus(Status.SUCCESS)
              .setExitCode(result.getExitCode())
              .build();
        } catch (CacheNotFoundException e) {
          // TODO(ulfjack): Track down who throws this exception in what cases and double-check that
          // ignoring the exception here is acceptable. Possible change it so that we throw in some
          // cases - we don't want to hide failures in the remote cache from the user.
        }
      }
      SpawnResult spawnResult = delegate.exec(spawn, policy);
      if (options.remoteUploadLocalResults
          && spawnResult.status() == Status.SUCCESS
          && spawnResult.exitCode() == 0) {
        writeCacheEntry(spawn, policy.getFileOutErr(), actionKey);
      }
      return spawnResult;
    } catch (StatusRuntimeException e) {
      throw new UserExecException(mnemonic + " remote work failed (" + e + ")", e);
    }
  }

  private Action buildAction(
      Collection<? extends ActionInput> outputs,
      Digest command,
      Digest inputRoot,
      Duration timeout) {
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
    if (!timeout.isZero()) {
      action.setTimeout(com.google.protobuf.Duration.newBuilder().setSeconds(timeout.getSeconds()));
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

  private void writeCacheEntry(Spawn spawn, FileOutErr outErr, ActionKey actionKey)
      throws IOException, InterruptedException {
    ArrayList<Path> outputFiles = new ArrayList<>();
    for (ActionInput output : spawn.getOutputFiles()) {
      Path outputPath = execRoot.getRelative(output.getExecPathString());
      // TODO(ulfjack): Store the actual list of output files in SpawnResult and use that instead
      // of statting the files here again.
      if (outputPath.exists()) {
        outputFiles.add(outputPath);
      }
    }
    remoteCache.upload(actionKey, execRoot, outputFiles, outErr);
  }
}
