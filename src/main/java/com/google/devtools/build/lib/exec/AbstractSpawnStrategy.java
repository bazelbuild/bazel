// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.exec;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.LostInputsActionExecutionException;
import com.google.devtools.build.lib.actions.LostInputsExecException;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.RunningActionEvent;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext;
import com.google.devtools.build.lib.actions.SchedulingActionEvent;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.exec.SpawnCache.CacheHandle;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.SortedMap;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/** Abstract common ancestor for spawn strategies implementing the common parts. */
public abstract class AbstractSpawnStrategy implements SandboxedSpawnActionContext {

  /**
   * Last unique identifier assigned to a spawn by this strategy.
   *
   * <p>These identifiers must be unique per strategy within the context of a Bazel server instance
   * to avoid cross-contamination across actions in case we perform asynchronous deletions.
   */
  private static final AtomicInteger execCount = new AtomicInteger();

  private final SpawnInputExpander spawnInputExpander;
  private final SpawnRunner spawnRunner;

  public AbstractSpawnStrategy(Path execRoot, SpawnRunner spawnRunner) {
    this.spawnInputExpander = new SpawnInputExpander(execRoot, false);
    this.spawnRunner = spawnRunner;
  }

  /**
   * Get's the {@link SpawnRunner} that this {@link AbstractSpawnStrategy} uses to actually run
   * spawns.
   *
   * <p>This is considered a stop-gap until we refactor the entire SpawnStrategy / SpawnRunner
   * mechanism to no longer need Spawn strategies.
   */
  public SpawnRunner getSpawnRunner() {
    return spawnRunner;
  }

  @Override
  public List<SpawnResult> exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    return exec(spawn, actionExecutionContext, null);
  }

  @Override
  public boolean canExec(Spawn spawn) {
    return spawnRunner.canExec(spawn);
  }

  @Override
  public List<SpawnResult> exec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      @Nullable StopConcurrentSpawns stopConcurrentSpawns)
      throws ExecException, InterruptedException {
    actionExecutionContext.maybeReportSubcommand(spawn);

    final Duration timeout = Spawns.getTimeout(spawn);
    SpawnExecutionContext context =
        new SpawnExecutionContextImpl(spawn, actionExecutionContext, stopConcurrentSpawns, timeout);
    // TODO(ulfjack): Provide a way to disable the cache. We don't want the RemoteSpawnStrategy to
    // check the cache twice. Right now that can't happen because this is hidden behind an
    // experimental flag.
    SpawnCache cache = actionExecutionContext.getContext(SpawnCache.class);
    // In production, the getContext method guarantees that we never get null back. However, our
    // integration tests don't set it up correctly, so cache may be null in testing.
    if (cache == null) {
      cache = SpawnCache.NO_CACHE;
    }
    SpawnResult spawnResult;
    ExecException ex = null;
    try (CacheHandle cacheHandle = cache.lookup(spawn, context)) {
      if (cacheHandle.hasResult()) {
        spawnResult = Preconditions.checkNotNull(cacheHandle.getResult());
      } else {
        // Actual execution.
        spawnResult = spawnRunner.execAsync(spawn, context).get();
        if (cacheHandle.willStore()) {
          cacheHandle.store(spawnResult);
        }
      }
    } catch (IOException e) {
      throw new EnvironmentalExecException(e);
    } catch (SpawnExecException e) {
      ex = e;
      spawnResult = e.getSpawnResult();
      // Log the Spawn and re-throw.
    }

    SpawnLogContext spawnLogContext = actionExecutionContext.getContext(SpawnLogContext.class);
    if (spawnLogContext != null) {
      try {
        spawnLogContext.logSpawn(
            spawn,
            actionExecutionContext.getMetadataProvider(),
            context.getInputMapping(true),
            context.getTimeout(),
            spawnResult);
      } catch (IOException e) {
        actionExecutionContext
            .getEventHandler()
            .handle(
                Event.warn("Exception " + e + " while logging properties of " + spawn.toString()));
      }
    }
    if (ex != null) {
      throw ex;
    }

    if (spawnResult.status() != Status.SUCCESS) {
      String cwd = actionExecutionContext.getExecRoot().getPathString();
      String resultMessage = spawnResult.getFailureMessage();
      String message =
          !Strings.isNullOrEmpty(resultMessage)
              ? resultMessage
              : CommandFailureUtils.describeCommandFailure(
                  actionExecutionContext.getVerboseFailures(),
                  spawn.getArguments(),
                  spawn.getEnvironment(),
                  cwd,
                  spawn.getExecutionPlatform());
      throw new SpawnExecException(message, spawnResult, /*forciblyRunRemotely=*/false);
    }
    return ImmutableList.of(spawnResult);
  }

  private final class SpawnExecutionContextImpl implements SpawnExecutionContext {
    private final Spawn spawn;
    private final ActionExecutionContext actionExecutionContext;
    @Nullable private final StopConcurrentSpawns stopConcurrentSpawns;
    private final Duration timeout;

    private final int id = execCount.incrementAndGet();
    // Memoize the input mapping so that prefetchInputs can reuse it instead of recomputing it.
    // TODO(ulfjack): Guard against client modification of this map.
    private SortedMap<PathFragment, ActionInput> lazyInputMapping;

    SpawnExecutionContextImpl(
        Spawn spawn,
        ActionExecutionContext actionExecutionContext,
        @Nullable StopConcurrentSpawns stopConcurrentSpawns,
        Duration timeout) {
      this.spawn = spawn;
      this.actionExecutionContext = actionExecutionContext;
      this.stopConcurrentSpawns = stopConcurrentSpawns;
      this.timeout = timeout;
    }

    @Override
    public int getId() {
      return id;
    }

    @Override
    public void prefetchInputs() throws IOException, InterruptedException {
      if (Spawns.shouldPrefetchInputsForLocalExecution(spawn)) {
        actionExecutionContext
            .getActionInputPrefetcher()
            .prefetchFiles(getInputMapping(true).values(), getMetadataProvider());
      }
    }

    @Override
    public MetadataProvider getMetadataProvider() {
      return actionExecutionContext.getMetadataProvider();
    }

    @Override
    public MetadataHandler getMetadataInjector() {
      return actionExecutionContext.getMetadataHandler();
    }

    @Override
    public ArtifactExpander getArtifactExpander() {
      return actionExecutionContext.getArtifactExpander();
    }

    @Override
    public ArtifactPathResolver getPathResolver() {
      return actionExecutionContext.getPathResolver();
    }

    @Override
    public void lockOutputFiles() throws InterruptedException {
      if (stopConcurrentSpawns != null) {
        stopConcurrentSpawns.stop();
      }
    }

    @Override
    public boolean speculating() {
      return stopConcurrentSpawns != null;
    }

    @Override
    public Duration getTimeout() {
      return timeout;
    }

    @Override
    public FileOutErr getFileOutErr() {
      return actionExecutionContext.getFileOutErr();
    }

    @Override
    public SortedMap<PathFragment, ActionInput> getInputMapping(
        boolean expandTreeArtifactsInRunfiles) throws IOException {
      if (lazyInputMapping == null) {
        try (SilentCloseable c =
            Profiler.instance().profile("AbstractSpawnStrategy.getInputMapping")) {
          lazyInputMapping =
              spawnInputExpander.getInputMapping(
                  spawn,
                  actionExecutionContext.getArtifactExpander(),
                  actionExecutionContext.getPathResolver(),
                  actionExecutionContext.getMetadataProvider(),
                  expandTreeArtifactsInRunfiles);
        }
      }
      return lazyInputMapping;
    }

    @Override
    public void report(ProgressStatus state, String name) {
      ActionExecutionMetadata action = spawn.getResourceOwner();
      if (action.getOwner() == null) {
        return;
      }

      // TODO(djasper): This should not happen as per the contract of ActionExecutionMetadata, but
      // there are implementations that violate the contract. Remove when those are gone.
      if (action.getPrimaryOutput() == null) {
        return;
      }

      // TODO(ulfjack): We should report more details to the UI.
      ExtendedEventHandler eventHandler = actionExecutionContext.getEventHandler();
      switch (state) {
        case EXECUTING:
        case CHECKING_CACHE:
          eventHandler.post(new RunningActionEvent(action, name));
          break;
        case SCHEDULING:
          eventHandler.post(new SchedulingActionEvent(action, name));
          break;
        default:
          break;
      }
    }

    @Override
    public void checkForLostInputs() throws LostInputsExecException {
      try {
        actionExecutionContext.checkForLostInputs();
      } catch (LostInputsActionExecutionException e) {
        throw e.toExecException();
      }
    }
  }
}
