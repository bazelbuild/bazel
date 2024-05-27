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
import static com.google.devtools.build.lib.profiler.ProfilerTask.REMOTE_DOWNLOAD;
import static com.google.devtools.build.lib.remote.util.Utils.createSpawnResult;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Stopwatch;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.ForbiddenActionInputException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.SpawnCache;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.RemoteExecutionService.InFlightExecution;
import com.google.devtools.build.lib.remote.RemoteExecutionService.RemoteActionResult;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.remote.util.Utils.InMemoryOutput;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentHashMap;

/** A remote {@link SpawnCache} implementation. */
@ThreadSafe // If the RemoteActionCache implementation is thread-safe.
final class RemoteSpawnCache implements SpawnCache {

  private final Path execRoot;
  private final RemoteOptions options;
  private final RemoteExecutionService remoteExecutionService;
  private final DigestUtil digestUtil;
  private final boolean verboseFailures;
  private final ConcurrentHashMap<RemoteCacheClient.ActionKey, InFlightExecution>
      inFlightExecutions = new ConcurrentHashMap<>();

  RemoteSpawnCache(
      Path execRoot,
      RemoteOptions options,
      boolean verboseFailures,
      RemoteExecutionService remoteExecutionService,
      DigestUtil digestUtil) {
    this.execRoot = execRoot;
    this.options = options;
    this.verboseFailures = verboseFailures;
    this.remoteExecutionService = remoteExecutionService;
    this.digestUtil = digestUtil;
  }

  @VisibleForTesting
  RemoteExecutionService getRemoteExecutionService() {
    return remoteExecutionService;
  }

  @Override
  public CacheHandle lookup(Spawn spawn, SpawnExecutionContext context)
      throws InterruptedException, IOException, ExecException, ForbiddenActionInputException {
    boolean shouldAcceptCachedResult =
        remoteExecutionService.getReadCachePolicy(spawn).allowAnyCache();
    boolean shouldUploadLocalResults =
        remoteExecutionService.getWriteCachePolicy(spawn).allowAnyCache();
    if (!shouldAcceptCachedResult && !shouldUploadLocalResults) {
      return SpawnCache.NO_RESULT_NO_STORE;
    }

    Stopwatch totalTime = Stopwatch.createStarted();

    RemoteAction action = remoteExecutionService.buildRemoteAction(spawn, context);
    SpawnMetrics.Builder spawnMetrics =
        SpawnMetrics.Builder.forRemoteExec()
            .setInputBytes(action.getInputBytes())
            .setInputFiles(action.getInputFiles());

    context.setDigest(digestUtil.asSpawnLogProto(action.getActionKey()));

    Profiler prof = Profiler.instance();
    InFlightExecution thisExecution = null;
    if (shouldAcceptCachedResult) {
      InFlightExecution firstExecution = null;
      thisExecution = InFlightExecution.createIfDeduplicatable(action);
      if (shouldUploadLocalResults && thisExecution != null) {
        firstExecution = inFlightExecutions.putIfAbsent(action.getActionKey(), thisExecution);
      }
      // Metadata will be available in context.current() until we detach.
      // This is done via a thread-local variable.
      try {
        RemoteActionResult result;
        try (SilentCloseable c = prof.profile(ProfilerTask.REMOTE_CACHE_CHECK, "check cache hit")) {
          result = remoteExecutionService.lookupCache(action);
        }
        // In case the remote cache returned a failed action (exit code != 0) we treat it as a
        // cache miss
        if (result != null && result.getExitCode() == 0) {
          Stopwatch fetchTime = Stopwatch.createStarted();
          InMemoryOutput inMemoryOutput;
          try (SilentCloseable c = prof.profile(REMOTE_DOWNLOAD, "download outputs")) {
            inMemoryOutput = remoteExecutionService.downloadOutputs(action, result);
          }
          fetchTime.stop();
          totalTime.stop();
          spawnMetrics
              .setFetchTimeInMs((int) fetchTime.elapsed().toMillis())
              .setTotalTimeInMs((int) totalTime.elapsed().toMillis())
              .setNetworkTimeInMs((int) action.getNetworkTime().getDuration().toMillis());
          SpawnResult spawnResult =
              createSpawnResult(
                  digestUtil,
                  action.getActionKey(),
                  result.getExitCode(),
                  /* cacheHit= */ true,
                  result.cacheName(),
                  inMemoryOutput,
                  result.getExecutionMetadata().getExecutionStartTimestamp(),
                  result.getExecutionMetadata().getExecutionCompletedTimestamp(),
                  spawnMetrics.build(),
                  spawn.getMnemonic());
          return SpawnCache.success(spawnResult);
        }
      } catch (CacheNotFoundException e) {
        // Intentionally left blank
      } catch (IOException e) {
        if (BulkTransferException.allCausedByCacheNotFoundException(e)) {
          // Intentionally left blank
        } else {
          String errorMessage = Utils.grpcAwareErrorMessage(e, verboseFailures);
          if (isNullOrEmpty(errorMessage)) {
            errorMessage = e.getClass().getSimpleName();
          }
          errorMessage = "Remote Cache: " + errorMessage;
          remoteExecutionService.report(Event.warn(errorMessage));
        }
      }
      if (firstExecution != null) {
        Stopwatch fetchTime = Stopwatch.createStarted();
        SpawnResult previousResult;
        try (SilentCloseable c = prof.profile(REMOTE_DOWNLOAD, "reuse outputs")) {
          previousResult = remoteExecutionService.reuseOutputs(action, firstExecution);
        }
        spawnMetrics
            .setFetchTimeInMs((int) fetchTime.elapsed().toMillis())
            .setTotalTimeInMs((int) totalTime.elapsed().toMillis())
            .setNetworkTimeInMs((int) action.getNetworkTime().getDuration().toMillis());
        SpawnMetrics buildMetrics = spawnMetrics.build();
        return SpawnCache.success(
            new SpawnResult.DelegateSpawnResult(previousResult) {
              @Override
              public String getRunnerName() {
                return "deduplicated";
              }

              @Override
              public SpawnMetrics getMetrics() {
                return buildMetrics;
              }
            });
      }
    }

    if (shouldUploadLocalResults) {
      final InFlightExecution thisExecutionFinal = thisExecution;
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
        public void store(SpawnResult result) throws ExecException, InterruptedException {
          if (!remoteExecutionService.shouldUpload(action, result, thisExecutionFinal)) {
            return;
          }

          if (options.experimentalGuardAgainstConcurrentChanges) {
            try (SilentCloseable c = prof.profile("RemoteCache.checkForConcurrentModifications")) {
              checkForConcurrentModifications();
            } catch (IOException | ForbiddenActionInputException e) {
              String msg =
                  "Skipping uploading outputs because of concurrent modifications "
                      + "with --experimental_guard_against_concurrent_changes enabled: "
                      + e.getMessage();
              remoteExecutionService.report(Event.warn(msg));
              return;
            }
          }

          remoteExecutionService.uploadOutputs(
              action, result, () -> inFlightExecutions.remove(action.getActionKey()));
        }

        private void checkForConcurrentModifications()
            throws IOException, ForbiddenActionInputException {
          for (ActionInput input : action.getInputMap(true).values()) {
            if (input instanceof VirtualActionInput) {
              continue;
            }
            FileArtifactValue metadata = context.getInputMetadataProvider().getInputMetadata(input);
            Path path = execRoot.getRelative(input.getExecPath());
            if (metadata.wasModifiedSinceDigest(path)) {
              throw new IOException(path + " was modified during thisExecution");
            }
          }
        }

        @Override
        public void reportException(Throwable e) {
          if (thisExecutionFinal != null) {
            thisExecutionFinal.reportExecutionException(e);
          }
        }
      };
    } else {
      return SpawnCache.NO_RESULT_NO_STORE;
    }
  }

  @Override
  public boolean usefulInDynamicExecution() {
    return false;
  }
}
