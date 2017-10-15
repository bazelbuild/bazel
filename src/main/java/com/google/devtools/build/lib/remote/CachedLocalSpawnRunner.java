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

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
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
import com.google.devtools.remoteexecution.v1test.Platform;
import java.io.IOException;
import java.util.List;
import java.util.SortedMap;

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
    this.platform = options.parseRemotePlatformOverride();
    this.remoteCache = remoteCache;
    this.delegate = delegate;
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionPolicy policy)
      throws InterruptedException, IOException, ExecException {

    // Temporary hack: the TreeNodeRepository should be created and maintained upstream!
    TreeNodeRepository repository =
        new TreeNodeRepository(execRoot, policy.getActionInputFileCache());
    SortedMap<PathFragment, ActionInput> inputMap = policy.getInputMapping();
    TreeNode inputRoot = repository.buildFromActionInputs(inputMap);
    repository.computeMerkleDigests(inputRoot);
    Command command =
        RemoteSpawnRunner.buildCommand(spawn.getArguments(), spawn.getEnvironment());
    Action action =
        RemoteSpawnRunner.buildAction(
            spawn.getOutputFiles(),
            Digests.computeDigest(command),
            repository.getMerkleDigest(inputRoot),
            platform,
            policy.getTimeout());

    // Look up action cache, and reuse the action output if it is found.
    ActionKey actionKey = Digests.computeActionKey(action);
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
        // There's a cache miss. Fall back to local execution.
      }
    }
    SpawnResult spawnResult = delegate.exec(spawn, policy);
    if (options.remoteUploadLocalResults
        && spawnResult.status() == Status.SUCCESS
        && spawnResult.exitCode() == 0) {
      List<Path> outputFiles = RemoteSpawnRunner.listExistingOutputFiles(execRoot, spawn);
      remoteCache.upload(actionKey, execRoot, outputFiles, policy.getFileOutErr());
    }
    return spawnResult;
  }
}
