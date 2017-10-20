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
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.SpawnCache;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.remote.Digests.ActionKey;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.remoteexecution.v1test.Action;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Command;
import com.google.devtools.remoteexecution.v1test.Platform;
import io.grpc.Context;
import java.io.IOException;
import java.util.Collection;
import java.util.NoSuchElementException;
import java.util.SortedMap;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

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
  private final String buildRequestId;
  private final String commandId;
  private final boolean verboseFailures;

  @Nullable private final Reporter cmdlineReporter;

  // Used to ensure that a warning is reported only once.
  private final AtomicBoolean warningReported = new AtomicBoolean();

  RemoteSpawnCache(
      Path execRoot,
      RemoteOptions options,
      RemoteActionCache remoteCache,
      String buildRequestId,
      String commandId,
      boolean verboseFailures,
      @Nullable Reporter cmdlineReporter) {
    this.execRoot = execRoot;
    this.options = options;
    this.platform = options.parseRemotePlatformOverride();
    this.remoteCache = remoteCache;
    this.verboseFailures = verboseFailures;
    this.cmdlineReporter = cmdlineReporter;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
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
    Context withMetadata =
        TracingMetadataUtils.contextWithMetadata(buildRequestId, commandId, actionKey);
    // Metadata will be available in context.current() until we detach.
    // This is done via a thread-local variable.
    Context previous = withMetadata.attach();
    try {
      ActionResult result =
          this.options.remoteAcceptCached ? remoteCache.getCachedActionResult(actionKey) : null;
      if (result != null) {
        // We don't cache failed actions, so we know the outputs exist.
        // For now, download all outputs locally; in the future, we can reuse the digests to
        // just update the TreeNodeRepository and continue the build.
        try {
          remoteCache.download(result, execRoot, policy.getFileOutErr());
          SpawnResult spawnResult =
              new SpawnResult.Builder()
                  .setStatus(Status.SUCCESS)
                  .setExitCode(result.getExitCode())
                  .build();
          return SpawnCache.success(spawnResult);
        } catch (CacheNotFoundException e) {
          // There's a cache miss. Fall back to local execution.
        }
      }
    } finally {
      withMetadata.detach(previous);
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
          boolean uploadAction = Status.SUCCESS.equals(result.status()) && result.exitCode() == 0;
          Context previous = withMetadata.attach();
          try {
            remoteCache.upload(actionKey, execRoot, files, policy.getFileOutErr(), uploadAction);
          } catch (IOException e) {
            if (verboseFailures) {
              report(Event.debug("Upload to remote cache failed: " + e.getMessage()));
            } else {
              reportOnce(Event.warn("Some artifacts failed be uploaded to the remote cache."));
            }
          } finally {
            withMetadata.detach(previous);
          }
        }

        @Override
        public void close() {}
      };
    } else {
      return SpawnCache.NO_RESULT_NO_STORE;
    }
  }

  private void reportOnce(Event evt) {
    if (warningReported.compareAndSet(false, true)) {
      report(evt);
    }
  }

  private void report(Event evt) {
    if (cmdlineReporter != null) {
      cmdlineReporter.handle(evt);
    }
  }
}
