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

import static com.google.common.base.Strings.isNullOrEmpty;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.SpawnCache;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOptions.FetchRemoteOutputsStrategy;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import io.grpc.Context;
import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.SortedMap;
import javax.annotation.Nullable;

/** A remote {@link SpawnCache} implementation. */
@ThreadSafe // If the RemoteActionCache implementation is thread-safe.
@ExecutionStrategy(
    name = {"remote-cache"},
    contextType = SpawnCache.class)
final class RemoteSpawnCache implements SpawnCache {
  private final Path execRoot;
  private final RemoteOptions options;

  private final AbstractRemoteActionCache remoteCache;
  private final String buildRequestId;
  private final String commandId;

  @Nullable private final Reporter cmdlineReporter;

  private final Set<String> reportedErrors = new HashSet<>();

  private final DigestUtil digestUtil;

  RemoteSpawnCache(
      Path execRoot,
      RemoteOptions options,
      AbstractRemoteActionCache remoteCache,
      String buildRequestId,
      String commandId,
      @Nullable Reporter cmdlineReporter,
      DigestUtil digestUtil) {
    this.execRoot = execRoot;
    this.options = options;
    this.remoteCache = remoteCache;
    this.cmdlineReporter = cmdlineReporter;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.digestUtil = digestUtil;
  }

  @Override
  public CacheHandle lookup(Spawn spawn, SpawnExecutionContext context)
      throws InterruptedException, IOException, ExecException {
    boolean checkCache = options.remoteAcceptCached && Spawns.mayBeCached(spawn);

    if (checkCache) {
      context.report(ProgressStatus.CHECKING_CACHE, "remote-cache");
    }

    SortedMap<PathFragment, ActionInput> inputMap = context.getInputMapping(true);
    // Temporary hack: the TreeNodeRepository should be created and maintained upstream!
    TreeNodeRepository repository =
        new TreeNodeRepository(
            execRoot,
            context.getMetadataProvider(),
            digestUtil,
            options.incompatibleRemoteSymlinks);
    TreeNode inputRoot;
    try (SilentCloseable c = Profiler.instance().profile("RemoteCache.computeMerkleDigests")) {
      inputRoot = repository.buildFromActionInputs(inputMap);
      repository.computeMerkleDigests(inputRoot);
    }
    Command command =
        RemoteSpawnRunner.buildCommand(
            spawn.getOutputFiles(),
            spawn.getArguments(),
            spawn.getEnvironment(),
            spawn.getExecutionPlatform());
    FetchRemoteOutputsStrategy remoteOutputsStrategy = options.experimentalRemoteFetchOutputs;
    Action action;
    final ActionKey actionKey;
    try (SilentCloseable c = Profiler.instance().profile("RemoteCache.buildAction")) {
      action =
          RemoteSpawnRunner.buildAction(
              digestUtil.compute(command),
              repository.getMerkleDigest(inputRoot),
              context.getTimeout(),
              Spawns.mayBeCached(spawn));
      // Look up action cache, and reuse the action output if it is found.
      actionKey = digestUtil.computeActionKey(action);
    }

    Context withMetadata =
        TracingMetadataUtils.contextWithMetadata(buildRequestId, commandId, actionKey);

    if (checkCache) {
      // Metadata will be available in context.current() until we detach.
      // This is done via a thread-local variable.
      Context previous = withMetadata.attach();
      try {
        ActionResult result;
        try (SilentCloseable c = Profiler.instance().profile("RemoteCache.getCachedActionResult")) {
          result = remoteCache.getCachedActionResult(actionKey);
        }
        if (result != null) {
          switch (remoteOutputsStrategy) {
            case MINIMAL:
              remoteCache.injectRemoteMetadata(result, spawn.getOutputFiles(),
                  context.getRequiredLocalOutputs(), context.getFileOutErr(), execRoot,
                  context.getMetadataInjector());
              break;
            case ALL:
              try (SilentCloseable c = Profiler.instance().profile("RemoteCache.download")) {
                remoteCache.download(result, execRoot, context.getFileOutErr());
              }
              break;
          }
          SpawnResult spawnResult = new SpawnResult.Builder()
              .setStatus(Status.SUCCESS)
              .setExitCode(result.getExitCode())
              .setCacheHit(true)
              .setRunnerName("remote cache hit")
              .build();
          return SpawnCache.success(spawnResult);
        }
      } catch (IOException e) {
        if (!AbstractRemoteActionCache.causedByCacheMiss(e)) {
          String errorMsg = e.getMessage();
          if (isNullOrEmpty(errorMsg)) {
            errorMsg = e.getClass().getSimpleName();
          }
          errorMsg = "Error reading from the remote cache:\n" + errorMsg;
          report(Event.warn(errorMsg));
        }
      } finally {
        withMetadata.detach(previous);
      }
    }

    context.prefetchInputs();

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
        public void store(SpawnResult result)
            throws ExecException, InterruptedException, IOException {
          if (options.experimentalGuardAgainstConcurrentChanges) {
            try (SilentCloseable c =
                Profiler.instance().profile("RemoteCache.checkForConcurrentModifications")) {
              checkForConcurrentModifications();
            } catch (IOException e) {
              report(Event.warn(e.getMessage()));
              return;
            }
          }
          boolean uploadAction =
              Spawns.mayBeCached(spawn)
                  && Status.SUCCESS.equals(result.status())
                  && result.exitCode() == 0;
          Context previous = withMetadata.attach();
          Collection<Path> files =
              RemoteSpawnRunner.resolveActionInputs(execRoot, spawn.getOutputFiles());
          try (SilentCloseable c = Profiler.instance().profile("RemoteCache.upload")) {
            remoteCache.upload(
                actionKey, action, command, execRoot, files, context.getFileOutErr(), uploadAction);
          } catch (IOException e) {
            String errorMsg = e.getMessage();
            if (isNullOrEmpty(errorMsg)) {
              errorMsg = e.getClass().getSimpleName();
            }
            errorMsg = "Error writing to the remote cache:\n" + errorMsg;
            report(Event.warn(errorMsg));
          } finally {
            withMetadata.detach(previous);
          }
        }

        @Override
        public void close() {}

        private void checkForConcurrentModifications() throws IOException {
          for (ActionInput input : inputMap.values()) {
            if (input instanceof VirtualActionInput) {
              continue;
            }
            FileArtifactValue metadata = context.getMetadataProvider().getMetadata(input);
            Path path = execRoot.getRelative(input.getExecPath());
            if (metadata.wasModifiedSinceDigest(path)) {
              throw new IOException(path + " was modified during execution");
            }
          }
        }
      };
    } else {
      return SpawnCache.NO_RESULT_NO_STORE;
    }
  }

  private void report(Event evt) {
    if (cmdlineReporter == null) {
      return;
    }

    synchronized (this) {
      if (reportedErrors.contains(evt.getMessage())) {
        return;
      }
      reportedErrors.add(evt.getMessage());
      cmdlineReporter.handle(evt);
    }
  }
}
