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

import com.google.common.base.Stopwatch;
import com.google.common.base.Throwables;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnMetrics;
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
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.RemoteExecutionService.RemoteAction;
import com.google.devtools.build.lib.remote.RemoteExecutionService.RemoteActionResult;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.remote.util.Utils.InMemoryOutput;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.HashSet;
import java.util.NoSuchElementException;
import java.util.Set;
import javax.annotation.Nullable;

/** A remote {@link SpawnCache} implementation. */
@ThreadSafe // If the RemoteActionCache implementation is thread-safe.
final class RemoteSpawnCache implements SpawnCache {

  private final Path execRoot;
  private final RemoteOptions options;
  private final boolean verboseFailures;
  @Nullable private final Reporter cmdlineReporter;
  private final Set<String> reportedErrors = new HashSet<>();
  private final RemoteExecutionService remoteExecutionService;

  RemoteSpawnCache(
      Path execRoot,
      RemoteOptions options,
      boolean verboseFailures,
      @Nullable Reporter cmdlineReporter,
      RemoteExecutionService remoteExecutionService) {
    this.execRoot = execRoot;
    this.options = options;
    this.verboseFailures = verboseFailures;
    this.cmdlineReporter = cmdlineReporter;
    this.remoteExecutionService = remoteExecutionService;
  }

  @Override
  public CacheHandle lookup(Spawn spawn, SpawnExecutionContext context)
      throws InterruptedException, IOException, ExecException {
    if (!Spawns.mayBeCached(spawn)
        || (!Spawns.mayBeCachedRemotely(spawn) && useRemoteCache(options))) {
      // returning SpawnCache.NO_RESULT_NO_STORE in case the caching is disabled or in case
      // the remote caching is disabled and the only configured cache is remote.
      return SpawnCache.NO_RESULT_NO_STORE;
    }

    Stopwatch totalTime = Stopwatch.createStarted();

    RemoteAction action = remoteExecutionService.buildRemoteAction(spawn, context);
    SpawnMetrics.Builder spawnMetrics =
        SpawnMetrics.Builder.forRemoteExec()
            .setInputBytes(action.getInputBytes())
            .setInputFiles(action.getInputFiles());

    Profiler prof = Profiler.instance();
    if (options.remoteAcceptCached
        || (options.incompatibleRemoteResultsIgnoreDisk && useDiskCache(options))) {
      context.report(ProgressStatus.CHECKING_CACHE, "remote-cache");
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
              .setFetchTime(fetchTime.elapsed())
              .setTotalTime(totalTime.elapsed())
              .setNetworkTime(action.getNetworkTime().getDuration());
          SpawnResult spawnResult =
              createSpawnResult(
                  result.getExitCode(),
                  /*cacheHit=*/ true,
                  "remote",
                  inMemoryOutput,
                  spawnMetrics.build(),
                  spawn.getMnemonic());
          return SpawnCache.success(spawnResult);
        }
      } catch (CacheNotFoundException e) {
        // Intentionally left blank
      } catch (IOException e) {
        if (BulkTransferException.isOnlyCausedByCacheNotFoundException(e)) {
          // Intentionally left blank
        } else {
          String errorMessage;
          if (!verboseFailures) {
            errorMessage = Utils.grpcAwareErrorMessage(e);
          } else {
            // On --verbose_failures print the whole stack trace
            errorMessage = Throwables.getStackTraceAsString(e);
          }
          if (isNullOrEmpty(errorMessage)) {
            errorMessage = e.getClass().getSimpleName();
          }
          errorMessage = "Reading from Remote Cache:\n" + errorMessage;
          report(Event.warn(errorMessage));
        }
      }
    }

    context.prefetchInputs();

    if (options.remoteUploadLocalResults
        || (options.incompatibleRemoteResultsIgnoreDisk && useDiskCache(options))) {
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
          boolean uploadResults = Status.SUCCESS.equals(result.status()) && result.exitCode() == 0;
          if (!uploadResults) {
            return;
          }

          if (options.experimentalGuardAgainstConcurrentChanges) {
            try (SilentCloseable c = prof.profile("RemoteCache.checkForConcurrentModifications")) {
              checkForConcurrentModifications();
            } catch (IOException e) {
              report(Event.warn(e.getMessage()));
              return;
            }
          }

          try (SilentCloseable c = prof.profile(ProfilerTask.UPLOAD_TIME, "upload outputs")) {
            remoteExecutionService.uploadOutputs(action);
          } catch (IOException e) {
            String errorMessage;
            if (!verboseFailures) {
              errorMessage = Utils.grpcAwareErrorMessage(e);
            } else {
              // On --verbose_failures print the whole stack trace
              errorMessage = Throwables.getStackTraceAsString(e);
            }
            if (isNullOrEmpty(errorMessage)) {
              errorMessage = e.getClass().getSimpleName();
            }
            errorMessage = "Writing to Remote Cache:\n" + errorMessage;
            report(Event.warn(errorMessage));
          }
        }

        @Override
        public void close() {}

        private void checkForConcurrentModifications() throws IOException {
          for (ActionInput input : action.getInputMap().values()) {
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

  private static boolean useRemoteCache(RemoteOptions options) {
    return !isNullOrEmpty(options.remoteCache) || !isNullOrEmpty(options.remoteExecutor);
  }

  private static boolean useDiskCache(RemoteOptions options) {
    return options.diskCache != null && !options.diskCache.isEmpty();
  }

  @Override
  public boolean usefulInDynamicExecution() {
    return false;
  }
}
