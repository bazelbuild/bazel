// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.devtools.build.lib.vfs.FileSystemUtils.createDirectoryAndParents;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.eventbus.EventBus;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionCacheChecker.Token;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionExecutedEvent;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContextFactory;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionLogBufferPathGenerator;
import com.google.devtools.build.lib.actions.ActionMiddlemanEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.AlreadyReportedActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ArtifactFile;
import com.google.devtools.build.lib.actions.ArtifactPrefixConflictException;
import com.google.devtools.build.lib.actions.CachedActionEvent;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.MapBasedActionGraph;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.actions.PackageRootResolutionException;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandle;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.TargetOutOfDateException;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.concurrent.Sharder;
import com.google.devtools.build.lib.concurrent.ThrowableRecordingRunnableWrapper;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.OutputService;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.protobuf.ByteString;

import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ConcurrentNavigableMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.FutureTask;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * Action executor: takes care of preparing an action for execution, executing it, validating that
 * all output artifacts were created, error reporting, etc.
 */
public final class SkyframeActionExecutor implements ActionExecutionContextFactory {
  private Reporter reporter;
  private final AtomicReference<EventBus> eventBus;
  private final ResourceManager resourceManager;
  private Executor executorEngine;
  private ActionLogBufferPathGenerator actionLogBufferPathGenerator;
  private ActionCacheChecker actionCacheChecker;
  private final Profiler profiler = Profiler.instance();
  private boolean explain;

  // We keep track of actions already executed this build in order to avoid executing a shared
  // action twice. Note that we may still unnecessarily re-execute the action on a subsequent
  // build: say actions A and B are shared. If A is requested on the first build and then B is
  // requested on the second build, we will execute B even though its output files are up to date.
  // However, we will not re-execute A on a subsequent build.
  // We do not allow the shared action to re-execute in the same build, even after the first
  // action has finished execution, because a downstream action might be reading the output file
  // at the same time as the shared action was writing to it.
  // This map is also used for Actions that try to execute twice because they have discovered
  // headers -- the SkyFunction tries to declare a dep on the missing headers and has to restart.
  // We don't want to execute the action again on the second entry to the SkyFunction.
  // In both cases, we store the already-computed ActionExecutionValue to avoid having to compute it
  // again.
  private ConcurrentMap<Artifact, Pair<Action, FutureTask<ActionExecutionValue>>> buildActionMap;

  // Errors found when examining all actions in the graph are stored here, so that they can be
  // thrown when execution of the action is requested. This field is set during each call to
  // findAndStoreArtifactConflicts, and is preserved across builds otherwise.
  private ImmutableMap<Action, ConflictException> badActionMap = ImmutableMap.of();
  private boolean keepGoing;
  private boolean hadExecutionError;
  private ActionInputFileCache perBuildFileCache;
  private ProgressSupplier progressSupplier;
  private ActionCompletedReceiver completionReceiver;
  private final AtomicReference<ActionExecutionStatusReporter> statusReporterRef;
  private OutputService outputService;

  SkyframeActionExecutor(ResourceManager resourceManager,
      AtomicReference<EventBus> eventBus,
      AtomicReference<ActionExecutionStatusReporter> statusReporterRef) {
    this.resourceManager = resourceManager;
    this.eventBus = eventBus;
    this.statusReporterRef = statusReporterRef;
  }

  /**
   * A typed union of {@link ActionConflictException}, which indicates two actions that generate
   * the same {@link Artifact}, and {@link ArtifactPrefixConflictException}, which indicates that
   * the path of one {@link Artifact} is a prefix of another.
   */
  public static class ConflictException extends Exception {
    @Nullable private final ActionConflictException ace;
    @Nullable private final ArtifactPrefixConflictException apce;

    public ConflictException(ActionConflictException e) {
      super(e);
      this.ace = e;
      this.apce = null;
    }

    public ConflictException(ArtifactPrefixConflictException e) {
      super(e);
      this.ace = null;
      this.apce = e;
    }

    void rethrowTyped() throws ActionConflictException, ArtifactPrefixConflictException {
      if (ace == null) {
        throw Preconditions.checkNotNull(apce);
      }
      if (apce == null) {
        throw Preconditions.checkNotNull(ace);
      }
      throw new IllegalStateException();
    }
  }

  /**
   * Return the map of mostly recently executed bad actions to their corresponding exception.
   * See {#findAndStoreArtifactConflicts()}.
   */
  public ImmutableMap<Action, ConflictException> badActions() {
    // TODO(bazel-team): Move badActions() and findAndStoreArtifactConflicts() to SkyframeBuildView
    // now that it's done in the analysis phase.
    return badActionMap;
  }

  /**
   * Find conflicts between generated artifacts. There are two ways to have conflicts. First, if
   * two (unshareable) actions generate the same output artifact, this will result in an {@link
   * ActionConflictException}. Second, if one action generates an artifact whose path is a prefix of
   * another artifact's path, those two artifacts cannot exist simultaneously in the output tree.
   * This causes an {@link ArtifactPrefixConflictException}. The relevant exceptions are stored in
   * the executor in {@code badActionMap}, and will be thrown immediately when that action is
   * executed. Those exceptions persist, so that even if the action is not executed this build, the
   * first time it is executed, the correct exception will be thrown.
   *
   * <p>This method must be called if a new action was added to the graph this build, so
   * whenever a new configured target was analyzed this build. It is somewhat expensive (~1s
   * range for a medium build as of 2014), so it should only be called when necessary.
   *
   * <p>Conflicts found may not be requested this build, and so we may overzealously throw an error.
   * For instance, if actions A and B generate the same artifact foo, and the user first requests
   * A' depending on A, and then in a subsequent build B' depending on B, we will fail the second
   * build, even though it would have succeeded if it had been the only build. However, since
   * Skyframe does not know the transitive dependencies of the request, we err on the conservative
   * side.
   *
   * <p>If the user first runs one action on the first build, and on the second build adds a
   * conflicting action, only the second action's error may be reported (because the first action
   * will be cached), whereas if both actions were requested for the first time, both errors would
   * be reported. However, the first time an action is added to the build, we are guaranteed to find
   * any conflicts it has, since this method will compare it against all other actions. So there is
   * no sequence of builds that can evade the error.
   */
  void findAndStoreArtifactConflicts(Iterable<ActionLookupValue> actionLookupValues)
      throws InterruptedException {
    ConcurrentMap<Action, ConflictException> temporaryBadActionMap = new ConcurrentHashMap<>();
    Pair<ActionGraph, SortedMap<PathFragment, Artifact>> result;
    result = constructActionGraphAndPathMap(actionLookupValues, temporaryBadActionMap);
    ActionGraph actionGraph = result.first;
    SortedMap<PathFragment, Artifact> artifactPathMap = result.second;

    // Report an error for every derived artifact which is a prefix of another.
    // If x << y << z (where x << y means "y starts with x"), then we only report (x,y), (x,z), but
    // not (y,z).
    Iterator<PathFragment> iter = artifactPathMap.keySet().iterator();
    if (!iter.hasNext()) {
      // No actions in graph -- currently happens only in tests. Special-cased because .next() call
      // below is unconditional.
      this.badActionMap = ImmutableMap.of();
      return;
    }
    for (PathFragment pathJ = iter.next(); iter.hasNext(); ) {
      // For each comparison, we have a prefix candidate (pathI) and a suffix candidate (pathJ).
      // At the beginning of the loop, we set pathI to the last suffix candidate, since it has not
      // yet been tested as a prefix candidate, and then set pathJ to the paths coming after pathI,
      // until we come to one that does not contain pathI as a prefix. pathI is then verified not to
      // be the prefix of any path, so we start the next run of the loop.
      PathFragment pathI = pathJ;
      // Compare pathI to the paths coming after it.
      while (iter.hasNext()) {
        pathJ = iter.next();
        if (pathJ.startsWith(pathI)) { // prefix conflict.
          Artifact artifactI = Preconditions.checkNotNull(artifactPathMap.get(pathI), pathI);
          Artifact artifactJ = Preconditions.checkNotNull(artifactPathMap.get(pathJ), pathJ);
          Action actionI =
              Preconditions.checkNotNull(actionGraph.getGeneratingAction(artifactI), artifactI);
          Action actionJ =
              Preconditions.checkNotNull(actionGraph.getGeneratingAction(artifactJ), artifactJ);
          if (actionI.shouldReportPathPrefixConflict(actionJ)) {
            ArtifactPrefixConflictException exception = new ArtifactPrefixConflictException(pathI,
                pathJ, actionI.getOwner().getLabel(), actionJ.getOwner().getLabel());
            temporaryBadActionMap.put(actionI, new ConflictException(exception));
            temporaryBadActionMap.put(actionJ, new ConflictException(exception));
          }
        } else { // pathJ didn't have prefix pathI, so no conflict possible for pathI.
          break;
        }
      }
    }
    this.badActionMap = ImmutableMap.copyOf(temporaryBadActionMap);
  }

  /**
   * Simultaneously construct an action graph for all the actions in Skyframe and a map from
   * {@link PathFragment}s to their respective {@link Artifact}s. We do this in a threadpool to save
   * around 1.5 seconds on a mid-sized build versus a single-threaded operation.
   */
  private static Pair<ActionGraph, SortedMap<PathFragment, Artifact>>
      constructActionGraphAndPathMap(
          Iterable<ActionLookupValue> values,
          ConcurrentMap<Action, ConflictException> badActionMap) throws InterruptedException {
    MutableActionGraph actionGraph = new MapBasedActionGraph();
    ConcurrentNavigableMap<PathFragment, Artifact> artifactPathMap = new ConcurrentSkipListMap<>();
    // Action graph construction is CPU-bound.
    int numJobs = Runtime.getRuntime().availableProcessors();
    // No great reason for expecting 5000 action lookup values, but not worth counting size of
    // values.
    Sharder<ActionLookupValue> actionShards = new Sharder<>(numJobs, 5000);
    for (ActionLookupValue value : values) {
      actionShards.add(value);
    }

    ThrowableRecordingRunnableWrapper wrapper = new ThrowableRecordingRunnableWrapper(
        "SkyframeActionExecutor#constructActionGraphAndPathMap");

    ExecutorService executor = Executors.newFixedThreadPool(
        numJobs,
        new ThreadFactoryBuilder().setNameFormat("ActionLookupValue Processor %d").build());
    for (List<ActionLookupValue> shard : actionShards) {
      executor.execute(
          wrapper.wrap(actionRegistration(shard, actionGraph, artifactPathMap, badActionMap)));
    }
    boolean interrupted = ExecutorUtil.interruptibleShutdown(executor);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted) {
      throw new InterruptedException();
    }
    return Pair.<ActionGraph, SortedMap<PathFragment, Artifact>>of(actionGraph, artifactPathMap);
  }

  private static Runnable actionRegistration(
      final List<ActionLookupValue> values,
      final MutableActionGraph actionGraph,
      final ConcurrentMap<PathFragment, Artifact> artifactPathMap,
      final ConcurrentMap<Action, ConflictException> badActionMap) {
    return new Runnable() {
      @Override
      public void run() {
        for (ActionLookupValue value : values) {
          Set<Action> registeredActions = new HashSet<>();
          for (Map.Entry<Artifact, Action> entry : value.getMapForConsistencyCheck().entrySet()) {
            Action action = entry.getValue();
            // We have an entry for each <action, artifact> pair. Only try to register each action
            // once.
            if (registeredActions.add(action)) {
              try {
                actionGraph.registerAction(action);
              } catch (ActionConflictException e) {
                Exception oldException = badActionMap.put(action, new ConflictException(e));
                Preconditions.checkState(oldException == null,
                  "%s | %s | %s", action, e, oldException);
                // We skip the rest of the loop, and do not add the path->artifact mapping for this
                // artifact below -- we don't need to check it since this action is already in
                // error.
                continue;
              }
            }
            artifactPathMap.put(entry.getKey().getExecPath(), entry.getKey());
          }
        }
      }
    };
  }

  void prepareForExecution(Reporter reporter, Executor executor, boolean keepGoing,
      boolean explain, ActionCacheChecker actionCacheChecker, OutputService outputService) {
    this.reporter = Preconditions.checkNotNull(reporter);
    this.executorEngine = Preconditions.checkNotNull(executor);

    // Start with a new map each build so there's no issue with internal resizing.
    this.buildActionMap = Maps.newConcurrentMap();
    this.keepGoing = keepGoing;
    this.hadExecutionError = false;
    this.actionCacheChecker = Preconditions.checkNotNull(actionCacheChecker);
    // Don't cache possibly stale data from the last build.
    this.explain = explain;
    this.outputService = outputService;
  }

  public void setActionLogBufferPathGenerator(
      ActionLogBufferPathGenerator actionLogBufferPathGenerator) {
    this.actionLogBufferPathGenerator = actionLogBufferPathGenerator;
  }

  void executionOver() {
    this.reporter = null;
    // This transitively holds a bunch of heavy objects, so it's important to clear it at the
    // end of a build.
    this.executorEngine = null;
    this.outputService = null;
  }

  boolean probeActionExecution(Action action) {
    return buildActionMap.containsKey(action.getPrimaryOutput());
  }

  private boolean actionReallyExecuted(Action action) {
    Pair<Action, ?> cachedRun = Preconditions.checkNotNull(
        buildActionMap.get(action.getPrimaryOutput()), action);
    return action == cachedRun.first;
  }

  /**
   * Executes the provided action on the current thread. Returns the ActionExecutionValue with the
   * result, either computed here or already computed on another thread.
   *
   * <p>For use from {@link ArtifactFunction} only.
   */
  ActionExecutionValue executeAction(Action action, ActionMetadataHandler metadataHandler,
      long actionStartTime,
      ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Exception exception = badActionMap.get(action);
    if (exception != null) {
      // If action had a conflict with some other action in the graph, report it now.
      reportError(exception.getMessage(), exception, action, null);
    }
    Artifact primaryOutput = action.getPrimaryOutput();
    FutureTask<ActionExecutionValue> actionTask =
        new FutureTask<>(new ActionRunner(action, metadataHandler,
            actionStartTime, actionExecutionContext));
    // Check to see if another action is already executing/has executed this value.
    Pair<Action, FutureTask<ActionExecutionValue>> oldAction =
        buildActionMap.putIfAbsent(primaryOutput, Pair.of(action, actionTask));

    if (oldAction == null) {
      actionTask.run();
    } else if (action == oldAction.first) {
      // We only allow the same action to be executed twice if it discovers inputs. We allow that
      // because we need to declare additional dependencies on those new inputs.
      Preconditions.checkState(action.discoversInputs(),
          "Same action shouldn't execute twice in build: %s", action);
      actionTask = oldAction.second;
    } else {
      Preconditions.checkState(Actions.canBeShared(oldAction.first, action),
          "Actions cannot be shared: %s %s", oldAction.first, action);
      // Wait for other action to finish, so any actions that depend on its outputs can execute.
      actionTask = oldAction.second;
    }
    try {
      return actionTask.get();
    } catch (ExecutionException e) {
      Throwables.propagateIfPossible(e.getCause(),
          ActionExecutionException.class, InterruptedException.class);
      throw new IllegalStateException(e);
    } finally {
      String message = action.getProgressMessage();
      if (message != null) {
        // Tell the receiver that the action has completed *before* telling the reporter.
        // This way the latter will correctly show the number of completed actions when task
        // completion messages are enabled (--show_task_finish).
        if (completionReceiver != null) {
          completionReceiver.actionCompleted(action);
        }
        reporter.finishTask(null, prependExecPhaseStats(message));
      }
    }
  }

  private static class ArtifactExpanderImpl implements ArtifactExpander {
    private final Map<Artifact, Collection<ArtifactFile>> expandedInputs;

    private ArtifactExpanderImpl(Map<Artifact, Collection<ArtifactFile>> expandedInputMiddlemen) {
      this.expandedInputs = expandedInputMiddlemen;
    }

    @Override
    public void expand(Artifact artifact, Collection<? super ArtifactFile> output) {
      Preconditions.checkState(artifact.isMiddlemanArtifact() || artifact.isTreeArtifact(),
          artifact);
      Collection<ArtifactFile> result = expandedInputs.get(artifact);
      // Note that result may be null for non-aggregating middlemen.
      if (result != null) {
        output.addAll(result);
      }
    }
  }

  /**
   * Returns an ActionExecutionContext suitable for executing a particular action. The caller should
   * pass the returned context to {@link #executeAction}, and any other method that needs to execute
   * tasks related to that action.
   */
  @Override
  public ActionExecutionContext getContext(
      ActionInputFileCache graphFileCache, MetadataHandler metadataHandler,
      Map<Artifact, Collection<ArtifactFile>> expandedInputs) {
    FileOutErr fileOutErr = actionLogBufferPathGenerator.generate();
    return new ActionExecutionContext(
        executorEngine,
        new DelegatingPairFileCache(graphFileCache, perBuildFileCache),
        metadataHandler,
        fileOutErr,
        new ArtifactExpanderImpl(expandedInputs));
  }

  /**
   * Checks the action cache to see if {@code action} needs to be executed, or is up to date.
   * Returns a token with the semantics of {@link ActionCacheChecker#getTokenIfNeedToExecute}: null
   * if the action is up to date, and non-null if it needs to be executed, in which case that token
   * should be provided to the ActionCacheChecker after execution.
   */
  Token checkActionCache(Action action, MetadataHandler metadataHandler,
      long actionStartTime, Iterable<Artifact> resolvedCacheArtifacts) {
    profiler.startTask(ProfilerTask.ACTION_CHECK, action);
    Token token = actionCacheChecker.getTokenIfNeedToExecute(
        action, resolvedCacheArtifacts, explain ? reporter : null, metadataHandler);
    profiler.completeTask(ProfilerTask.ACTION_CHECK);
    if (token == null) {
      boolean eventPosted = false;
      // Notify BlazeRuntimeStatistics about the action middleman 'execution'.
      if (action.getActionType().isMiddleman()) {
        postEvent(new ActionMiddlemanEvent(action, actionStartTime));
        eventPosted = true;
      }

      if (action instanceof NotifyOnActionCacheHit) {
        NotifyOnActionCacheHit notify = (NotifyOnActionCacheHit) action;
        notify.actionCacheHit(executorEngine);
      }

      // We still need to check the outputs so that output file data is available to the value.
      checkOutputs(action, metadataHandler);
      if (!eventPosted) {
        postEvent(new CachedActionEvent(action, actionStartTime));
      }
    }
    return token;
  }

  void afterExecution(Action action, MetadataHandler metadataHandler, Token token) {
    if (!actionReallyExecuted(action)) {
      // If an action shared with this one executed, then we need not update the action cache, since
      // the other action will do it. Moreover, this action is not aware of metadata acquired
      // during execution, so its metadata handler is likely unusable anyway.
      return;
    }
    try {
      actionCacheChecker.afterExecution(action, token, metadataHandler);
    } catch (IOException e) {
      // Skyframe has already done all the filesystem access needed for outputs and swallows
      // IOExceptions for inputs. So an IOException is impossible here.
      throw new IllegalStateException(
          "failed to update action cache for " + action.prettyPrint()
              + ", but all outputs should already have been checked", e);
    }
  }

  @Nullable
  Iterable<Artifact> getActionCachedInputs(Action action, PackageRootResolver resolver)
      throws PackageRootResolutionException {
    return actionCacheChecker.getCachedInputs(action, resolver);
  }

  /**
   * Perform dependency discovery for action, which must discover its inputs.
   *
   * <p>This method is just a wrapper around {@link Action#discoverInputs} that properly processes
   * any ActionExecutionException thrown before rethrowing it to the caller.
   */
  Collection<Artifact> discoverInputs(Action action, PerActionFileCache graphFileCache,
      MetadataHandler metadataHandler, Environment env)
      throws ActionExecutionException, InterruptedException {
    ActionExecutionContext actionExecutionContext =
        ActionExecutionContext.forInputDiscovery(
            executorEngine,
            new DelegatingPairFileCache(graphFileCache, perBuildFileCache),
            metadataHandler,
            actionLogBufferPathGenerator.generate(),
            env);
    try {
      return action.discoverInputs(actionExecutionContext);
    } catch (ActionExecutionException e) {
      throw processAndThrow(e, action, actionExecutionContext.getFileOutErr());
    }
  }

  /**
   * This method should be called if the builder encounters an error during
   * execution. This allows the builder to record that it encountered at
   * least one error, and may make it swallow its output to prevent
   * spamming the user any further.
   */
  private void recordExecutionError() {
    hadExecutionError = true;
  }

  /**
   * Returns true if the Builder is winding down (i.e. cancelling outstanding
   * actions and preparing to abort.)
   * The builder is winding down iff:
   * <ul>
   * <li>we had an execution error
   * <li>we are not running with --keep_going
   * </ul>
   */
  private boolean isBuilderAborting() {
    return hadExecutionError && !keepGoing;
  }

  void setFileCache(ActionInputFileCache fileCache) {
    this.perBuildFileCache = fileCache;
  }

  private class ActionRunner implements Callable<ActionExecutionValue> {
    private final Action action;
    private final ActionMetadataHandler metadataHandler;
    private long actionStartTime;
    private ActionExecutionContext actionExecutionContext;

    ActionRunner(Action action, ActionMetadataHandler metadataHandler,
        long actionStartTime,
        ActionExecutionContext actionExecutionContext) {
      this.action = action;
      this.metadataHandler = metadataHandler;
      this.actionStartTime = actionStartTime;
      this.actionExecutionContext = actionExecutionContext;
    }

    @Override
    public ActionExecutionValue call() throws ActionExecutionException, InterruptedException {
      profiler.startTask(ProfilerTask.ACTION, action);
      try {
        if (actionCacheChecker.isActionExecutionProhibited(action)) {
          // We can't execute an action (e.g. because --check_???_up_to_date option was used). Fail
          // the build instead.
          synchronized (reporter) {
            TargetOutOfDateException e = new TargetOutOfDateException(action);
            reporter.handle(Event.error(e.getMessage()));
            recordExecutionError();
            throw e;
          }
        }

        String message = action.getProgressMessage();
        if (message != null) {
          reporter.startTask(null, prependExecPhaseStats(message));
        }
        statusReporterRef.get().setPreparing(action);

        Preconditions.checkState(actionExecutionContext.getMetadataHandler() == metadataHandler,
            "%s %s", actionExecutionContext.getMetadataHandler(), metadataHandler);
        prepareScheduleExecuteAndCompleteAction(action, actionExecutionContext, actionStartTime);
        return new ActionExecutionValue(
            metadataHandler.getOutputArtifactFileData(),
            metadataHandler.getOutputTreeArtifactData(),
            metadataHandler.getAdditionalOutputData());
      } finally {
        profiler.completeTask(ProfilerTask.ACTION);
      }
    }
  }

  private void createOutputDirectories(Action action) throws ActionExecutionException {
    try {
      Set<Path> done = new HashSet<>(); // avoid redundant calls for the same directory.
      for (Artifact outputFile : action.getOutputs()) {
        Path outputDir;
        if (outputFile.isTreeArtifact()) {
          outputDir = outputFile.getPath();
        } else {
          outputDir = outputFile.getPath().getParentDirectory();
        }

        if (done.add(outputDir)) {
          try {
            createDirectoryAndParents(outputDir);
            continue;
          } catch (IOException e) {
            /* Fall through to plan B. */
          }

          // Possibly some direct ancestors are not directories.  In that case, we unlink all the
          // ancestors until we reach a directory, then try again. This handles the case where a
          // file becomes a directory, either from one build to another, or within a single build.
          //
          // Symlinks should not be followed so in order to clean up symlinks pointing to Fileset
          // outputs from previous builds. See bug [incremental build of Fileset fails if
          // Fileset.out was changed to be a subdirectory of the old value].
          try {
            for (Path p = outputDir; !p.isDirectory(Symlinks.NOFOLLOW);
                p = p.getParentDirectory()) {
              // p may be a file or dangling symlink, or a symlink to an old Fileset output
              p.delete(); // throws IOException
            }
            createDirectoryAndParents(outputDir);
          } catch (IOException e) {
            throw new ActionExecutionException(
                "failed to create output directory '" + outputDir + "'", e, action, false);
          }
        }
      }
    } catch (ActionExecutionException ex) {
      printError(ex.getMessage(), action, null);
      throw ex;
    }
  }

  private String prependExecPhaseStats(String message) {
    if (progressSupplier != null) {
      // Prints a progress message like:
      //   [2608/6445] Compiling foo/bar.cc [host]
      return progressSupplier.getProgressString() + " " + message;
    } else {
      // progressSupplier may be null in tests
      return message;
    }
  }

  /**
   * Prepare, schedule, execute, and then complete the action.
   * When this function is called, we know that this action needs to be executed.
   * This function will prepare for the action's execution (i.e. delete the outputs);
   * schedule its execution; execute the action;
   * and then do some post-execution processing to complete the action:
   * set the outputs readonly and executable, and insert the action results in the
   * action cache.
   *
   * @param action  The action to execute
   * @param context services in the scope of the action
   * @param actionStartTime time when we started the first phase of the action execution.
   * @throws ActionExecutionException if the execution of the specified action
   *   failed for any reason.
   * @throws InterruptedException if the thread was interrupted.
   */
  private void prepareScheduleExecuteAndCompleteAction(Action action,
      ActionExecutionContext context, long actionStartTime)
      throws ActionExecutionException, InterruptedException {
    // Delete the metadataHandler's cache of the action's outputs, since they are being deleted.
    context.getMetadataHandler().discardOutputMetadata();
    // Delete the outputs before executing the action, just to ensure that
    // the action really does produce the outputs.
    try {
      action.prepare(context.getExecutor().getExecRoot());
      createOutputDirectories(action);
    } catch (IOException e) {
      reportError("failed to delete output files before executing action", e, action, null);
    }

    postEvent(new ActionStartedEvent(action, actionStartTime));
    ResourceSet estimate =
        Preconditions.checkNotNull(action.estimateResourceConsumption(executorEngine));
    ActionExecutionStatusReporter statusReporter = statusReporterRef.get();
    ResourceHandle handle = null;
    try {
      if (estimate == ResourceSet.ZERO) {
        statusReporter.setRunningFromBuildData(action);
      } else {
        // If estimated resource consumption is null, action will manually call
        // resource manager when it knows what resources are needed.
        handle = resourceManager.acquireResources(action, estimate);
      }
      boolean outputDumped = executeActionTask(action, context);
      completeAction(action, context.getMetadataHandler(),
          context.getFileOutErr(), outputDumped);
    } finally {
      if (handle != null) {
        handle.close();
      }
      statusReporter.remove(action);
      postEvent(new ActionCompletionEvent(actionStartTime, action));
    }
  }

  ActionExecutionException processAndThrow(
      ActionExecutionException e, Action action, FileOutErr outErrBuffer)
      throws ActionExecutionException {
    reportActionExecution(action, e, outErrBuffer);
    boolean reported = reportErrorIfNotAbortingMode(e, outErrBuffer);

    ActionExecutionException toThrow = e;
    if (reported){
      // If we already printed the error for the exception we mark it as already reported
      // so that we do not print it again in upper levels.
      // Note that we need to report it here since we want immediate feedback of the errors
      // and in some cases the upper-level printing mechanism only prints one of the errors.
      toThrow = new AlreadyReportedActionExecutionException(e);
    }

    // Now, rethrow the exception.
    // This can have two effects:
    // If we're still building, the exception will get retrieved by the
    // completor and rethrown.
    // If we're aborting, the exception will never be retrieved from the
    // completor, since the completor is waiting for all outstanding jobs
    // to finish. After they have finished, it will only rethrow the
    // exception that initially caused it to abort will and not check the
    // exit status of any actions that had finished in the meantime.
    throw toThrow;
  }

  /**
   * Execute the specified action, in a profiler task.
   * The caller is responsible for having already checked that we need to
   * execute it and for acquiring/releasing any scheduling locks needed.
   *
   * <p>This is thread-safe so long as you don't try to execute the same action
   * twice at the same time (or overlapping times).
   * May execute in a worker thread.
   *
   * @throws ActionExecutionException if the execution of the specified action
   *   failed for any reason.
   * @throws InterruptedException if the thread was interrupted.
   * @return true if the action output was dumped, false otherwise.
   */
  private boolean executeActionTask(Action action, ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    profiler.startTask(ProfilerTask.ACTION_EXECUTE, action);
    // ActionExecutionExceptions that occur as the thread is interrupted are
    // assumed to be a result of that, so we throw InterruptedException
    // instead.
    FileOutErr outErrBuffer = actionExecutionContext.getFileOutErr();
    try {
      action.execute(actionExecutionContext);

      // Action terminated fine, now report the output.
      // The .showOutput() method is not necessarily a quick check: in its
      // current implementation it uses regular expression matching.
      if (outErrBuffer.hasRecordedOutput()
          && (action.showsOutputUnconditionally()
          || reporter.showOutput(Label.print(action.getOwner().getLabel())))) {
        dumpRecordedOutErr(action, outErrBuffer);
        return true;
      }
      // Defer reporting action success until outputs are checked
    } catch (ActionExecutionException e) {
      processAndThrow(e, action, outErrBuffer);
    } finally {
      profiler.completeTask(ProfilerTask.ACTION_EXECUTE);
    }
    return false;
  }

  private void completeAction(Action action, MetadataHandler metadataHandler,
      FileOutErr fileOutErr, boolean outputAlreadyDumped) throws ActionExecutionException {
    try {
      Preconditions.checkState(action.inputsKnown(),
          "Action %s successfully executed, but inputs still not known", action);

      profiler.startTask(ProfilerTask.ACTION_COMPLETE, action);
      try {
        if (!checkOutputs(action, metadataHandler)) {
          reportError("not all outputs were created", null, action,
              outputAlreadyDumped ? null : fileOutErr);
        }
        // Prevent accidental stomping on files.
        // This will also throw a FileNotFoundException
        // if any of the output files doesn't exist.
        try {
          setOutputsReadOnlyAndExecutable(action, metadataHandler);
        } catch (IOException e) {
          reportError("failed to set outputs read-only", e, action, null);
        }
      } finally {
        profiler.completeTask(ProfilerTask.ACTION_COMPLETE);
      }

      if (outputService != null) {
        try {
          outputService.finalizeAction(action, metadataHandler);
        } catch (EnvironmentalExecException | IOException e) {
          reportError("unable to finalize action: " + e.getMessage(), e, action, fileOutErr);
        }
      }

      reportActionExecution(action, null, fileOutErr);
    } catch (ActionExecutionException actionException) {
      // Success in execution but failure in completion.
      reportActionExecution(action, actionException, fileOutErr);
      throw actionException;
    } catch (IllegalStateException exception) {
      // More serious internal error, but failure still reported.
      reportActionExecution(action,
          new ActionExecutionException(exception, action, true), fileOutErr);
      throw exception;
    }
  }

  private static void setPathReadOnlyAndExecutable(MetadataHandler metadataHandler,
      ArtifactFile file)
      throws IOException {
    // If the metadata was injected, we assume the mode is set correct and bail out early to avoid
    // the additional overhead of resetting it.
    if (metadataHandler.isInjected(file)) {
      return;
    }
    Path path = file.getPath();
    if (path.isFile(Symlinks.NOFOLLOW)) { // i.e. regular files only.
      // We trust the files created by the execution-engine to be non symlinks with expected
      // chmod() settings already applied.
      path.chmod(0555);  // Sets the file read-only and executable.
    }
  }

  private static void setTreeReadOnlyAndExecutable(MetadataHandler metadataHandler, Artifact parent,
      PathFragment subpath) throws IOException {
    Path path = parent.getPath().getRelative(subpath);
    if (path.isDirectory()) {
      path.chmod(0555);
      for (Path child : path.getDirectoryEntries()) {
        setTreeReadOnlyAndExecutable(metadataHandler, parent,
            subpath.getChild(child.getBaseName()));
      }
    } else {
      setPathReadOnlyAndExecutable(
          metadataHandler, ActionInputHelper.artifactFile(parent, subpath));
    }
  }

  /**
   * For each of the action's outputs that is a regular file (not a symbolic
   * link or directory), make it read-only and executable.
   *
   * <p>Making the outputs read-only helps preventing accidental editing of
   * them (e.g. in case of generated source code), while making them executable
   * helps running generated files (such as generated shell scripts) on the
   * command line.
   *
   * <p>May execute in a worker thread.
   *
   * <p>Note: setting these bits maintains transparency regarding the locality of the build;
   * because the remote execution engine sets them, they should be set for local builds too.
   *
   * @throws IOException if an I/O error occurred.
   */
  private final void setOutputsReadOnlyAndExecutable(Action action, MetadataHandler metadataHandler)
      throws IOException {
    Preconditions.checkState(!action.getActionType().isMiddleman());

    for (Artifact output : action.getOutputs()) {
      if (output.isTreeArtifact()) {
        // Preserve existing behavior: we don't set non-TreeArtifact directories
        // read only and executable. However, it's unusual for non-TreeArtifact outputs
        // to be directories.
        setTreeReadOnlyAndExecutable(metadataHandler, output, PathFragment.EMPTY_FRAGMENT);
      } else {
        setPathReadOnlyAndExecutable(metadataHandler, output);
      }
    }
  }

  private void reportMissingOutputFile(Action action, Artifact output, Reporter reporter,
      boolean isSymlink) {
    boolean genrule = action.getMnemonic().equals("Genrule");
    String prefix = (genrule ? "declared output '" : "output '") + output.prettyPrint() + "' ";
    if (isSymlink) {
      reporter.handle(Event.error(
          action.getOwner().getLocation(), prefix + "is a dangling symbolic link"));
    } else {
      String suffix = genrule ? " by genrule. This is probably "
          + "because the genrule actually didn't create this output, or because the output was a "
          + "directory and the genrule was run remotely (note that only the contents of "
          + "declared file outputs are copied from genrules run remotely)" : "";
      reporter.handle(Event.error(
          action.getOwner().getLocation(), prefix + "was not created" + suffix));
    }
  }

  /**
   * Validates that all action outputs were created or intentionally omitted.
   *
   * @return false if some outputs are missing, true - otherwise.
   */
  private boolean checkOutputs(Action action, MetadataHandler metadataHandler) {
    boolean success = true;
    for (Artifact output : action.getOutputs()) {
      // artifactExists has the side effect of potentially adding the artifact to the cache,
      // therefore we only call it if we know the artifact is indeed not omitted to avoid any
      // unintended side effects.
      if (!(metadataHandler.artifactOmitted(output) || metadataHandler.artifactExists(output))) {
        reportMissingOutputFile(action, output, reporter, output.getPath().isSymbolicLink());
        success = false;
      }
    }
    return success;
  }

  private void postEvent(Object event) {
    EventBus bus = eventBus.get();
    if (bus != null) {
      bus.post(event);
    }
  }

  /**
   * Convenience function for reporting that the action failed due to a
   * the exception cause, if there is an additional explanatory message that
   * clarifies the message of the exception. Combines the user-provided message
   * and the exceptions' message and reports the combination as error.
   * Then, throws an ActionExecutionException with the reported error as
   * message and the provided exception as the cause.
   *
   * @param message A small text that explains why the action failed
   * @param cause The exception that caused the action to fail
   * @param action The action that failed
   * @param actionOutput The output of the failed Action.
   *     May be null, if there is no output to display
   */
  private void reportError(String message, Throwable cause, Action action, FileOutErr actionOutput)
      throws ActionExecutionException {
    ActionExecutionException ex;
    if (cause == null) {
      ex = new ActionExecutionException(message, action, false);
    } else {
      ex = new ActionExecutionException(message, cause, action, false);
    }
    printError(ex.getMessage(), action, actionOutput);
    throw ex;
  }

  /**
   * For the action 'action' that failed due to 'ex' with the output
   * 'actionOutput', notify the user about the error. To notify the user, the
   * method first displays the output of the action and then reports an error
   * via the reporter. The method ensures that the two messages appear next to
   * each other by locking the outErr object where the output is displayed.
   *
   * @param message The reason why the action failed
   * @param action The action that failed, must not be null.
   * @param actionOutput The output of the failed Action.
   *     May be null, if there is no output to display
   */
  private void printError(String message, Action action, FileOutErr actionOutput) {
    synchronized (reporter) {
      if (keepGoing) {
        message = "Couldn't " + describeAction(action) + ": " + message;
      }
      Event event = Event.error(action.getOwner().getLocation(), message);
      dumpRecordedOutErr(event, actionOutput);
      recordExecutionError();
    }
  }

  /** Describe an action, for use in error messages. */
  private static String describeAction(Action action) {
    if (action.getOutputs().isEmpty()) {
      return "run " + action.prettyPrint();
    } else if (action.getActionType().isMiddleman()) {
      return "build " + action.prettyPrint();
    } else {
      return "build file " + action.getPrimaryOutput().prettyPrint();
    }
  }

  /**
   * Dump the output from the action.
   *
   * @param action The action whose output is being dumped
   * @param outErrBuffer The OutErr that recorded the actions output
   */
  private void dumpRecordedOutErr(Action action, FileOutErr outErrBuffer) {
    StringBuilder message = new StringBuilder("");
    message.append("From ");
    message.append(action.describe());
    message.append(":");
    Event event = Event.info(message.toString());
    dumpRecordedOutErr(event, outErrBuffer);
  }

  /**
   * Dump the output from the action.
   *
   * @param prefixEvent An event to post before dumping the output
   * @param outErrBuffer The OutErr that recorded the actions output
   */
  private void dumpRecordedOutErr(Event prefixEvent, FileOutErr outErrBuffer) {
    // Synchronize this on the reporter, so that the output from multiple
    // actions will not be interleaved.
    synchronized (reporter) {
      // Only print the output if we're not winding down.
      if (isBuilderAborting()) {
        return;
      }
      reporter.handle(prefixEvent);

      if (outErrBuffer != null && outErrBuffer.hasRecordedOutput()) {
        OutErr outErr = this.reporter.getOutErr();
        outErrBuffer.dumpOutAsLatin1(outErr.getOutputStream());
        outErrBuffer.dumpErrAsLatin1(outErr.getErrorStream());
      }
    }
  }

  private void reportActionExecution(Action action,
      ActionExecutionException exception, FileOutErr outErr) {
    String stdout = null;
    String stderr = null;

    if (outErr.hasRecordedStdout()) {
      stdout = outErr.getOutputFile().toString();
    }
    if (outErr.hasRecordedStderr()) {
      stderr = outErr.getErrorFile().toString();
    }
    postEvent(new ActionExecutedEvent(action, exception, stdout, stderr));
  }

  /**
   * Returns true if the exception was reported. False otherwise. Currently this is a copy of what
   * we did in pre-Skyframe execution. The main implication is that we are printing the error to the
   * top level reporter instead of the action reporter. Because of that Skyframe values do not know
   * about the errors happening in the execution phase. Even if we change in the future to log to
   * the action reporter (that would be done in ActionExecutionFunction.compute() when we get an
   * ActionExecutionException), we probably do not want to also store the StdErr output, so
   * dumpRecordedOutErr() should still be called here.
   */
  private boolean reportErrorIfNotAbortingMode(ActionExecutionException ex,
      FileOutErr outErrBuffer) {
    // For some actions (e.g. many local actions) the pollInterruptedStatus()
    // won't notice that we had an interrupted job. It will continue.
    // For that reason we must take care to NOT report errors if we're
    // in the 'aborting' mode: Any cancelled action would show up here.
    // For some actions (e.g. many local actions) the pollInterruptedStatus()
    // won't notice that we had an interrupted job. It will continue.
    // For that reason we must take care to NOT report errors if we're
    // in the 'aborting' mode: Any cancelled action would show up here.
    synchronized (this.reporter) {
      if (!isBuilderAborting()) {
        // Oops. The action aborted. Report the problem.
        printError(ex.getMessage(), ex.getAction(), outErrBuffer);
        return true;
      }
    }
    return false;
  }

  /** An object supplying data for action execution progress reporting. */
  public interface ProgressSupplier {
    /** Returns the progress string to prefix action execution messages with. */
    String getProgressString();
  }

  /** An object that can be notified about action completion. */
  public interface ActionCompletedReceiver {
    /** Receives a completed action. */
    void actionCompleted(Action action);
  }

  public void setActionExecutionProgressReportingObjects(
      @Nullable ProgressSupplier progressSupplier,
      @Nullable ActionCompletedReceiver completionReceiver) {
    this.progressSupplier = progressSupplier;
    this.completionReceiver = completionReceiver;
  }

  private static class DelegatingPairFileCache implements ActionInputFileCache {
    private final ActionInputFileCache perActionCache;
    private final ActionInputFileCache perBuildFileCache;

    private DelegatingPairFileCache(ActionInputFileCache mainCache,
        ActionInputFileCache perBuildFileCache) {
      this.perActionCache = mainCache;
      this.perBuildFileCache = perBuildFileCache;
    }

    @Override
    public ByteString getDigest(ActionInput actionInput) throws IOException {
      ByteString digest = perActionCache.getDigest(actionInput);
      return digest != null ? digest : perBuildFileCache.getDigest(actionInput);
    }

    @Override
    public boolean isFile(Artifact input) {
      // PerActionCache must have a value for all artifacts.
      return perActionCache.isFile(input);
    }

    @Override
    public long getSizeInBytes(ActionInput actionInput) throws IOException {
      long size = perActionCache.getSizeInBytes(actionInput);
      return size > -1 ? size : perBuildFileCache.getSizeInBytes(actionInput);
    }

    @Override
    public boolean contentsAvailableLocally(ByteString digest) {
      return perActionCache.contentsAvailableLocally(digest)
          || perBuildFileCache.contentsAvailableLocally(digest);
    }

    @Nullable
    @Override
    public ActionInput getInputFromDigest(ByteString digest) throws IOException {
      ActionInput file = perActionCache.getInputFromDigest(digest);
      return file != null ? file : perBuildFileCache.getInputFromDigest(digest);
    }

    @Override
    public Path getInputPath(ActionInput input) {
      if (input instanceof Artifact) {
        return perActionCache.getInputPath(input);
      } else {
        return perBuildFileCache.getInputPath(input);
      }
    }
  }
}
