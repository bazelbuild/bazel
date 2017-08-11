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
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.exec.SpawnCache;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.remoteexecution.v1test.Action;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Command;
import com.google.devtools.remoteexecution.v1test.Platform;
import java.io.IOException;
import java.util.Collection;
import java.util.NoSuchElementException;
import java.util.SortedMap;

/**
 * A remote {@link SpawnCache} implementation.
 */
@ThreadSafe // If the RemoteActionCache implementation is thread-safe.
@ExecutionStrategy(
  name = {"remote-cache"},
  contextType = SpawnCache.class
)
final class RemoteSpawnCache implements SpawnCache {
  private final Path execRoot;
  private final RemoteOptions options;
  // TODO(olaola): This will be set on a per-action basis instead.
  private final Platform platform;

  private final RemoteActionCache remoteCache;

  RemoteSpawnCache(Path execRoot, RemoteOptions options, RemoteActionCache remoteCache) {
    this.execRoot = execRoot;
    this.options = options;
    this.platform = options.parseRemotePlatformOverride();
    this.remoteCache = remoteCache;
  }

  @Override
  public CacheHandle lookup(Spawn spawn, SpawnExecutionPolicy policy)
      throws InterruptedException, IOException, ExecException {
    // Temporary hack: the TreeNodeRepository should be created and maintained upstream!
    TreeNodeRepository repository =
        new TreeNodeRepository(execRoot, policy.getActionInputFileCache());
    SortedMap<PathFragment, ActionInput> inputMap = policy.getInputMapping();
    TreeNode inputRoot = repository.buildFromActionInputs(inputMap);
    repository.computeMerkleDigests(inputRoot);
    Command command = RemoteSpawnRunner.buildCommand(spawn.getArguments(), spawn.getEnvironment());
    Action action =
        RemoteSpawnRunner.buildAction(
            spawn.getOutputFiles(),
            Digests.computeDigest(command),
            repository.getMerkleDigest(inputRoot),
            platform,
            policy.getTimeout());

    // Look up action cache, and reuse the action output if it is found.
    final ActionKey actionKey = Digests.computeActionKey(action);
    ActionResult result =
        this.options.remoteAcceptCached ? remoteCache.getCachedActionResult(actionKey) : null;
    if (result != null) {
      // We don't cache failed actions, so we know the outputs exist.
      // For now, download all outputs locally; in the future, we can reuse the digests to
      // just update the TreeNodeRepository and continue the build.
      try {
        remoteCache.download(result, execRoot, policy.getFileOutErr());
        SpawnResult spawnResult = new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setExitCode(result.getExitCode())
            .build();
        return SpawnCache.success(spawnResult);
      } catch (CacheNotFoundException e) {
        // There's a cache miss. Fall back to local execution.
      }
    }
    if (options.remoteUploadLocalResults) {
      return new CacheHandle() {
        @Override
        public boolean hasResult() {
          return false;
        }

        @Override
        public SpawnResult getResult() {
          throw new NoSuchElementException();
        }

        @Override
        public boolean willStore() {
          return true;
        }

        @Override
        public void store(SpawnResult result, Collection<Path> files)
            throws InterruptedException, IOException {
          if (result.status() != Status.SUCCESS || result.exitCode() != 0) {
            return;
          }
          remoteCache.upload(actionKey, execRoot, files, policy.getFileOutErr());
        }

        @Override
        public void close() {
        }
      };
    } else {
      return SpawnCache.NO_RESULT_NO_STORE;
    }
  }
}
