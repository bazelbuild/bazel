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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.remote.ContentDigests.ActionKey;
import com.google.devtools.build.lib.remote.RemoteProtocol.Action;
import com.google.devtools.build.lib.remote.RemoteProtocol.ActionResult;
import com.google.devtools.build.lib.remote.RemoteProtocol.Command;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.remote.RemoteProtocol.Platform;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
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
import java.util.SortedMap;
import java.util.TreeSet;

/**
 * A {@link SpawnRunner} implementation that adds a remote cache on top of an underlying local
 * {@link SpawnRunner} implementation.
 */
final class CachedLocalSpawnRunner implements SpawnRunner {
  private final Path execRoot;
  private final RemoteOptions options;
  // TODO(olaola): This will be set on a per-action basis instead.
  private final Platform platform;

  private final RemoteActionCache actionCache;
  private final SpawnRunner delegate;

  CachedLocalSpawnRunner(
      Path execRoot,
      RemoteOptions options,
      RemoteActionCache actionCache,
      SpawnRunner delegate) {
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
    this.actionCache = actionCache;
    this.delegate = delegate;
  }

  @Override
  public SpawnResult exec(
      Spawn spawn,
      SpawnExecutionPolicy policy)
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
              ContentDigests.computeDigest(command),
              repository.getMerkleDigest(inputRoot));

      // Look up action cache, and reuse the action output if it is found.
      actionKey = ContentDigests.computeActionKey(action);
      ActionResult result =
          this.options.remoteAcceptCached ? actionCache.getCachedActionResult(actionKey) : null;
      if (result != null) {
        // We don't cache failed actions, so we know the outputs exist.
        // For now, download all outputs locally; in the future, we can reuse the digests to
        // just update the TreeNodeRepository and continue the build.
        try {
          // TODO(ulfjack): Download stdout, stderr, and the output files in a single call.
          actionCache.downloadAllResults(result, execRoot);
          passRemoteOutErr(result, policy.getFileOutErr());
          return new SpawnResult.Builder()
              .setSetupSuccess(true)
              .setExitCode(result.getReturnCode())
              .build();
        } catch (CacheNotFoundException e) {
          // TODO(ulfjack): Track down who throws this exception in what cases and double-check that
          // ignoring the exception here is acceptable. Possible change it so that we throw in some
          // cases - we don't want to hide failures in the remote cache from the user.
        }
      }
      SpawnResult spawnResult = delegate.exec(spawn, policy);
      if (options.remoteLocalExecUploadResults && spawnResult.setupSuccess()) {
        writeCacheEntry(spawn, actionKey);
      }
      return spawnResult;
    } catch (StatusRuntimeException e) {
      throw new UserExecException(mnemonic + " remote work failed (" + e + ")", e);
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

  private static Command buildCommand(
      List<String> arguments, ImmutableMap<String, String> environment) {
    Command.Builder command = Command.newBuilder();
    command.addAllArgv(arguments);
    // Sorting the environment pairs by variable name.
    TreeSet<String> variables = new TreeSet<>(environment.keySet());
    for (String var : variables) {
      command.addEnvironmentBuilder().setVariable(var).setValue(environment.get(var));
    }
    return command.build();
  }

  private void passRemoteOutErr(
      ActionResult result, FileOutErr outErr)
          throws CacheNotFoundException {
    ImmutableList<byte[]> streams =
        actionCache.downloadBlobs(
            ImmutableList.of(result.getStdoutDigest(), result.getStderrDigest()));
    outErr.printOut(new String(streams.get(0), UTF_8));
    outErr.printErr(new String(streams.get(1), UTF_8));
  }

  private void writeCacheEntry(Spawn spawn, ActionKey actionKey)
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
    ActionResult.Builder result = ActionResult.newBuilder();
    actionCache.uploadAllResults(execRoot, outputFiles, result);
    actionCache.setCachedActionResult(actionKey, result.build());
  }
}
