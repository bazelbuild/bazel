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

import com.google.common.base.Preconditions;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.Striped;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionCacheChecker.Token;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionContinuationOrResult;
import com.google.devtools.build.lib.actions.ActionExecutedEvent;
import com.google.devtools.build.lib.actions.ActionExecutedEvent.ErrorTiming;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.LostInputsCheck;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLogBufferPathGenerator;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionMiddlemanEvent;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.ActionResultReceivedEvent;
import com.google.devtools.build.lib.actions.ActionScanningCompletedEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.AlreadyReportedActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpanderImpl;
import com.google.devtools.build.lib.actions.Artifact.OwnerlessArtifactWrapper;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.CachedActionEvent;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.LostInputsActionExecutionException;
import com.google.devtools.build.lib.actions.MetadataConsumer;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit.ActionCachedContext;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.actions.ScanningActionEvent;
import com.google.devtools.build.lib.actions.SpawnResult.MetadataLog;
import com.google.devtools.build.lib.actions.StoppedScanningActionEvent;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetExpander;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.rules.cpp.IncludeScannable;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.ActionExecutionState.ActionStep;
import com.google.devtools.build.lib.skyframe.ActionExecutionState.ActionStepOrResult;
import com.google.devtools.build.lib.skyframe.ActionExecutionState.SharedActionCallback;
import com.google.devtools.build.lib.util.CrashFailureDetails;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.OutputService.ActionFileSystemType;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.common.options.OptionsProvider;
import java.io.Closeable;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Action executor: takes care of preparing an action for execution, executing it, validating that
 * all output artifacts were created, error reporting, etc.
 */
public final class SkyframeActionExecutor {
  static boolean actionDependsOnBuildId(Action action) {
    // Volatile build actions may need to execute even if none of their known inputs have changed.
    // Depending on the build id ensures that these actions have a chance to execute.
    // SkyframeAwareActions do not need to depend on the build id because their volatility is due to
    // their dependence on Skyframe nodes that are not captured in the action cache. Any changes to
    // those nodes will cause this action to be rerun, so a build id dependency is unnecessary.
    return (action.isVolatile() && !(action instanceof SkyframeAwareAction))
        || action instanceof NotifyOnActionCacheHit;
  }

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  // Used to prevent check-then-act races in #createOutputDirectories. See the comment there for
  // more detail.
  private static final Striped<Lock> outputDirectoryDeletionLock = Striped.lock(64);

  private final ActionKeyContext actionKeyContext;
  private Reporter reporter;
  private Map<String, String> clientEnv = ImmutableMap.of();
  private Executor executorEngine;
  private ExtendedEventHandler progressSuppressingEventHandler;
  private ActionLogBufferPathGenerator actionLogBufferPathGenerator;
  private ActionCacheChecker actionCacheChecker;
  @Nullable private TopDownActionCache topDownActionCache;
  private final Profiler profiler = Profiler.instance();

  // We keep track of actions already executed this build in order to avoid executing a shared
  // action twice. Note that we may still unnecessarily re-execute the action on a subsequent
  // build: say actions A and B are shared. If A is requested on the first build and then B is
  // requested on the second build, we will execute B even though its output files are up to date.
  // However, we will not re-execute A on a subsequent build.
  // We do not allow the shared action to re-execute in the same build, even after the first
  // action has finished execution, because a downstream action might be reading the output file
  // at the same time as the shared action was writing to it.
  //
  // This map is also used for Actions that try to execute twice because they have discovered
  // headers -- the SkyFunction tries to declare a dep on the missing headers and has to restart.
  // We don't want to execute the action again on the second entry to the SkyFunction.
  // In both cases, we store the already-computed ActionExecutionValue to avoid having to compute it
  // again.
  private ConcurrentMap<OwnerlessArtifactWrapper, ActionExecutionState> buildActionMap;

  // We also keep track of actions which were reset this build from a previously-completed state.
  // When re-evaluated, these actions should not emit ProgressLike events, in order to not confuse
  // the downstream consumers of action-related event streams, which may (reasonably) have expected
  // an action to be executed at most once per build.
  //
  // Note: actions which fail due to lost inputs, and get rewound, will not have any events
  // suppressed during their second evaluation. Consumers of events which get emitted before
  // execution (e.g. ActionStartedEvent, SpawnExecutedEvent) must support receiving more than one of
  // those events per action.
  private Set<OwnerlessArtifactWrapper> completedAndResetActions;

  // We also keep track of actions that failed due to lost discovered inputs. In some circumstances
  // the input discovery process will use a discovered input before requesting it as a dep. If that
  // input was generated but is lost, and action rewinding resets it and its generating action, then
  // the lost input's generating action must be rerun before the failed action tries input discovery
  // again. A previously failed action satisfies that requirement by requesting the deps in this map
  // at the start of its next attempt,
  private ConcurrentMap<OwnerlessArtifactWrapper, ImmutableList<SkyKey>> lostDiscoveredInputsMap;

  // Directories which are known to be created as regular directories within this invocation. This
  // implies parent directories are also regular directories.
  private Set<PathFragment> knownRegularDirectories;

  private OptionsProvider options;
  private boolean useAsyncExecution;
  private boolean hadExecutionError;
  private boolean replayActionOutErr;
  private MetadataProvider perBuildFileCache;
  private ActionInputPrefetcher actionInputPrefetcher;
  /** These variables are nulled out between executions. */
  private ProgressSupplier progressSupplier;
  private ActionCompletedReceiver completionReceiver;

  private final AtomicReference<ActionExecutionStatusReporter> statusReporterRef;
  private OutputService outputService;
  private boolean finalizeActions;
  private final Supplier<ImmutableList<Root>> sourceRootSupplier;

  private boolean bazelRemoteExecutionEnabled;

  private NestedSetExpander nestedSetExpander;

  SkyframeActionExecutor(
      ActionKeyContext actionKeyContext,
      AtomicReference<ActionExecutionStatusReporter> statusReporterRef,
      Supplier<ImmutableList<Root>> sourceRootSupplier) {
    this.actionKeyContext = actionKeyContext;
    this.statusReporterRef = statusReporterRef;
    this.sourceRootSupplier = sourceRootSupplier;
  }

  SharedActionCallback getSharedActionCallback(
      ExtendedEventHandler eventHandler,
      boolean hasDiscoveredInputs,
      Action action,
      ActionLookupData actionLookupData) {
    return new SharedActionCallback() {
      @Override
      public void actionStarted() {
        if (hasDiscoveredInputs) {
          eventHandler.post(new ActionScanningCompletedEvent(action, actionLookupData));
        }
      }

      @Override
      public void actionCompleted() {
        completionReceiver.actionCompleted(actionLookupData);
      }
    };
  }

  void prepareForExecution(
      Reporter reporter,
      Executor executor,
      OptionsProvider options,
      ActionCacheChecker actionCacheChecker,
      TopDownActionCache topDownActionCache,
      OutputService outputService) {
    this.reporter = Preconditions.checkNotNull(reporter);
    this.executorEngine = Preconditions.checkNotNull(executor);
    this.progressSuppressingEventHandler = new ProgressSuppressingEventHandler(reporter);

    // Start with a new map each build so there's no issue with internal resizing.
    this.buildActionMap = Maps.newConcurrentMap();
    this.completedAndResetActions = Sets.newConcurrentHashSet();
    this.lostDiscoveredInputsMap = Maps.newConcurrentMap();
    this.hadExecutionError = false;
    this.actionCacheChecker = Preconditions.checkNotNull(actionCacheChecker);
    this.topDownActionCache = topDownActionCache;
    // Don't cache possibly stale data from the last build.
    this.options = options;
    // Cache some option values for performance, since we consult them on every action.
    this.useAsyncExecution = options.getOptions(BuildRequestOptions.class).useAsyncExecution;
    this.finalizeActions = options.getOptions(BuildRequestOptions.class).finalizeActions;
    this.replayActionOutErr = options.getOptions(BuildRequestOptions.class).replayActionOutErr;
    this.outputService = outputService;
    RemoteOptions remoteOptions = options.getOptions(RemoteOptions.class);
    this.bazelRemoteExecutionEnabled = remoteOptions != null && remoteOptions.isRemoteEnabled();

    Cache<PathFragment, Boolean> cache =
        CacheBuilder.from(options.getOptions(BuildRequestOptions.class).directoryCreationCacheSpec)
            .concurrencyLevel(Runtime.getRuntime().availableProcessors())
            .build();
    this.knownRegularDirectories = Collections.newSetFromMap(cache.asMap());
  }

  public void setActionLogBufferPathGenerator(
      ActionLogBufferPathGenerator actionLogBufferPathGenerator) {
    this.actionLogBufferPathGenerator = actionLogBufferPathGenerator;
  }

  public void setClientEnv(Map<String, String> clientEnv) {
    // Copy once here, instead of on every construction of ActionExecutionContext.
    this.clientEnv = ImmutableMap.copyOf(clientEnv);
  }

  ActionFileSystemType actionFileSystemType() {
    return outputService != null
        ? outputService.actionFileSystemType()
        : ActionFileSystemType.DISABLED;
  }

  Path getExecRoot() {
    return executorEngine.getExecRoot();
  }

  /** REQUIRES: {@link #actionFileSystemType()} to be not {@code DISABLED}. */
  FileSystem createActionFileSystem(
      String relativeOutputPath,
      ActionInputMap inputArtifactData,
      Iterable<Artifact> outputArtifacts,
      boolean trackFailedRemoteReads) {
    return outputService.createActionFileSystem(
        executorEngine.getFileSystem(),
        executorEngine.getExecRoot().asFragment(),
        relativeOutputPath,
        sourceRootSupplier.get(),
        inputArtifactData,
        outputArtifacts,
        trackFailedRemoteReads);
  }

  private void updateActionFileSystemContext(
      Action action,
      FileSystem actionFileSystem,
      Environment env,
      MetadataConsumer consumer,
      ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> filesets)
      throws ActionExecutionException {
    try {
      outputService.updateActionFileSystemContext(actionFileSystem, env, consumer, filesets);
    } catch (IOException e) {
      String message = "Failed to update filesystem context: " + e.getMessage();
      DetailedExitCode code =
          createDetailedExitCode(message, Code.FILESYSTEM_CONTEXT_UPDATE_FAILURE);
      throw new ActionExecutionException(message, e, action, /*catastrophe=*/ false, code);
    }
  }

  void executionOver() {
    // These may transitively holds a bunch of heavy objects, so it's important to clear it at the
    // end of a build.
    this.reporter = null;
    this.options = null;
    this.executorEngine = null;
    this.progressSuppressingEventHandler = null;
    this.outputService = null;
    this.buildActionMap = null;
    this.completedAndResetActions = null;
    this.lostDiscoveredInputsMap = null;
    this.actionCacheChecker = null;
    this.topDownActionCache = null;
    this.knownRegularDirectories = null;
  }

  /**
   * Due to multi-threading, a null return value from this method does not guarantee that there is
   * no such action - a concurrent thread may already be executing the same (shared) action. Any
   * such race is resolved in the subsequent call to {@link #executeAction}.
   */
  @Nullable
  ActionExecutionState probeActionExecution(Action action) {
    return buildActionMap.get(new OwnerlessArtifactWrapper(action.getPrimaryOutput()));
  }

  /**
   * Determines whether the action should have its progress events emitted.
   *
   * <p>Returns {@code false} for completed and reset actions, indicating that their progress events
   * should be suppressed.
   */
  boolean shouldEmitProgressEvents(Action action) {
    return !completedAndResetActions.contains(
        new OwnerlessArtifactWrapper(action.getPrimaryOutput()));
  }

  void resetPreviouslyCompletedActionExecution(Action action) {
    OwnerlessArtifactWrapper ownerlessArtifactWrapper =
        new OwnerlessArtifactWrapper(action.getPrimaryOutput());
    buildActionMap.remove(ownerlessArtifactWrapper);
    completedAndResetActions.add(ownerlessArtifactWrapper);
  }

  @Nullable
  ImmutableList<SkyKey> getLostDiscoveredInputs(Action action) {
    return lostDiscoveredInputsMap.get(new OwnerlessArtifactWrapper(action.getPrimaryOutput()));
  }

  void resetFailedActionExecution(Action action, ImmutableList<SkyKey> lostDiscoveredInputs) {
    OwnerlessArtifactWrapper ownerlessArtifactWrapper =
        new OwnerlessArtifactWrapper(action.getPrimaryOutput());
    buildActionMap.remove(ownerlessArtifactWrapper);
    if (!lostDiscoveredInputs.isEmpty()) {
      lostDiscoveredInputsMap.put(ownerlessArtifactWrapper, lostDiscoveredInputs);
    }
  }

  void noteActionEvaluationStarted(ActionLookupData actionLookupData, Action action) {
    this.completionReceiver.noteActionEvaluationStarted(actionLookupData, action);
  }

  /**
   * Executes the provided action on the current thread. Returns the ActionExecutionValue with the
   * result, either computed here or already computed on another thread.
   */
  @SuppressWarnings("SynchronizeOnNonFinalField")
  ActionExecutionValue executeAction(
      Environment env,
      Action action,
      ActionMetadataHandler metadataHandler,
      long actionStartTime,
      ActionLookupData actionLookupData,
      Map<Artifact, Collection<Artifact>> expandedInputs,
      ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets,
      ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets,
      @Nullable FileSystem actionFileSystem,
      @Nullable Object skyframeDepsResult,
      ActionPostprocessing postprocessing,
      boolean hasDiscoveredInputs)
      throws ActionExecutionException, InterruptedException {
    MetadataAggregator metadataAggregator;
    if (actionFileSystem != null) {
      metadataAggregator = new MetadataAggregator(metadataHandler);
      updateActionFileSystemContext(
          action, actionFileSystem, env, metadataAggregator, expandedFilesets);
    } else {
      metadataAggregator = null;
    }

    ActionExecutionContext actionExecutionContext =
        getContext(
            env,
            action,
            metadataHandler,
            expandedInputs,
            expandedFilesets,
            topLevelFilesets,
            actionFileSystem,
            skyframeDepsResult);

    if (actionCacheChecker.isActionExecutionProhibited(action)) {
      // We can't execute an action (e.g. because --check_???_up_to_date option was used). Fail the
      // build instead.
      String message = action.prettyPrint() + " is not up-to-date";
      DetailedExitCode code = createDetailedExitCode(message, Code.ACTION_NOT_UP_TO_DATE);
      ActionExecutionException e = new ActionExecutionException(message, action, false, code);
      Event error = Event.error(e.getMessage());
      synchronized (reporter) {
        reporter.handle(error);
      }
      recordExecutionError();
      throw e;
    }

    // Use computeIfAbsent to handle concurrent attempts to execute the same shared action.
    ActionExecutionState activeAction =
        buildActionMap.computeIfAbsent(
            new OwnerlessArtifactWrapper(action.getPrimaryOutput()),
            (_unused_key) ->
                new ActionExecutionState(
                    actionLookupData,
                    new ActionRunner(
                        action,
                        metadataHandler,
                        actionStartTime,
                        actionExecutionContext,
                        actionLookupData,
                        postprocessing,
                        metadataAggregator)));

    SharedActionCallback callback =
        getSharedActionCallback(env.getListener(), hasDiscoveredInputs, action, actionLookupData);

    ActionExecutionValue result = null;
    ActionExecutionException finalException = null;
    try {
      result = activeAction.getResultOrDependOnFuture(env, actionLookupData, action, callback);
    } catch (ActionExecutionException e) {
      finalException = e;
    }

    if (result != null || finalException != null) {
      closeContext(actionExecutionContext, action, finalException);
    }
    return result;
  }

  private ExtendedEventHandler selectEventHandler(Action action) {
    return selectEventHandler(shouldEmitProgressEvents(action));
  }

  private ExtendedEventHandler selectEventHandler(boolean emitProgressEvents) {
    return emitProgressEvents ? reporter : progressSuppressingEventHandler;
  }

  TopDownActionCache getTopDownActionCache() {
    return topDownActionCache;
  }

  private ActionExecutionContext getContext(
      Environment env,
      Action action,
      MetadataHandler metadataHandler,
      Map<Artifact, Collection<Artifact>> expandedInputs,
      ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets,
      ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets,
      @Nullable FileSystem actionFileSystem,
      @Nullable Object skyframeDepsResult) {
    boolean emitProgressEvents = shouldEmitProgressEvents(action);
    ArtifactPathResolver artifactPathResolver =
        ArtifactPathResolver.createPathResolver(actionFileSystem, executorEngine.getExecRoot());
    FileOutErr fileOutErr;
    if (replayActionOutErr) {
      String actionKey = action.getKey(actionKeyContext, /*artifactExpander=*/ null);
      fileOutErr = actionLogBufferPathGenerator.persistent(actionKey, artifactPathResolver);
      try {
        fileOutErr.getErrorPath().delete();
        fileOutErr.getOutputPath().delete();
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
    } else {
      fileOutErr = actionLogBufferPathGenerator.generate(artifactPathResolver);
    }
    return new ActionExecutionContext(
        executorEngine,
        createFileCache(metadataHandler, actionFileSystem),
        actionInputPrefetcher,
        actionKeyContext,
        metadataHandler,
        env.restartPermitted(),
        lostInputsCheck(actionFileSystem, action, outputService),
        fileOutErr,
        replayActionOutErr && emitProgressEvents
            ? env.getListener()
            : selectEventHandler(emitProgressEvents),
        clientEnv,
        topLevelFilesets,
        new ArtifactExpanderImpl(expandedInputs, expandedFilesets),
        actionFileSystem,
        skyframeDepsResult,
        nestedSetExpander);
  }

  private static void closeContext(
      ActionExecutionContext context,
      Action action,
      @Nullable ActionExecutionException finalException)
      throws ActionExecutionException {
    try (Closeable c = context) {
      if (finalException != null) {
        throw finalException;
      }
    } catch (IOException e) {
      String message = "Failed to close action output: " + e.getMessage();
      DetailedExitCode code = createDetailedExitCode(message, Code.ACTION_OUTPUT_CLOSE_FAILURE);
      throw new ActionExecutionException(message, e, action, /*catastrophe=*/ false, code);
    }
  }

  /**
   * Checks the action cache to see if {@code action} needs to be executed, or is up to date.
   * Returns a token with the semantics of {@link ActionCacheChecker#getTokenIfNeedToExecute}: null
   * if the action is up to date, and non-null if it needs to be executed, in which case that token
   * should be provided to the ActionCacheChecker after execution.
   */
  Token checkActionCache(
      ExtendedEventHandler eventHandler,
      Action action,
      MetadataHandler metadataHandler,
      long actionStartTime,
      List<Artifact> resolvedCacheArtifacts,
      Map<String, String> clientEnv,
      ArtifactPathResolver pathResolver)
      throws ActionExecutionException {
    Token token;
    try (SilentCloseable c = profiler.profile(ProfilerTask.ACTION_CHECK, action.describe())) {
      RemoteOptions remoteOptions = this.options.getOptions(RemoteOptions.class);
      SortedMap<String, String> remoteDefaultProperties =
          remoteOptions != null
              ? remoteOptions.getRemoteDefaultExecProperties()
              : ImmutableSortedMap.of();
      token =
          actionCacheChecker.getTokenIfNeedToExecute(
              action,
              resolvedCacheArtifacts,
              clientEnv,
              options.getOptions(BuildRequestOptions.class).explanationPath != null
                  ? reporter
                  : null,
              metadataHandler,
              remoteDefaultProperties);
    } catch (UserExecException e) {
      throw e.toActionExecutionException(action);
    }
    if (token == null) {
      boolean eventPosted = false;
      // Notify BlazeRuntimeStatistics about the action middleman 'execution'.
      if (action.getActionType().isMiddleman()) {
        eventHandler.post(new ActionMiddlemanEvent(action, actionStartTime));
        eventPosted = true;
      }

      if (replayActionOutErr) {
        // TODO(ulfjack): This assumes that the stdout/stderr files are unmodified. It would be
        //  better to integrate them with the action cache and rerun the action when they change.
        String actionKey = action.getKey(actionKeyContext, /*artifactExpander=*/ null);
        FileOutErr fileOutErr = actionLogBufferPathGenerator.persistent(actionKey, pathResolver);
        // Set the mightHaveOutput bit in FileOutErr. Otherwise hasRecordedOutput() doesn't check if
        // the file exists and just returns false.
        fileOutErr.getOutputPath();
        fileOutErr.getErrorPath();
        if (fileOutErr.hasRecordedOutput()) {
          dumpRecordedOutErr(eventHandler, action, fileOutErr);
        }
      }

      if (action instanceof NotifyOnActionCacheHit) {
        NotifyOnActionCacheHit notify = (NotifyOnActionCacheHit) action;
        ExtendedEventHandler contextEventHandler = selectEventHandler(action);
        ActionCachedContext context =
            new ActionCachedContext() {
              @Override
              public ExtendedEventHandler getEventHandler() {
                return contextEventHandler;
              }

              @Override
              public Path getExecRoot() {
                return executorEngine.getExecRoot();
              }

              @Override
              public <T extends ActionContext> T getContext(Class<? extends T> type) {
                return executorEngine.getContext(type);
              }
            };
        notify.actionCacheHit(context);
      }

      // We still need to check the outputs so that output file data is available to the value.
      checkOutputs(action, metadataHandler);
      if (!eventPosted) {
        eventHandler.post(new CachedActionEvent(action, actionStartTime));
      }
    }
    return token;
  }

  void updateActionCache(
      Action action, MetadataHandler metadataHandler, Token token, Map<String, String> clientEnv)
      throws ActionExecutionException {
    if (!actionCacheChecker.enabled()) {
      return;
    }
    final SortedMap<String, String> remoteDefaultProperties;
    try {
      RemoteOptions remoteOptions = this.options.getOptions(RemoteOptions.class);
      remoteDefaultProperties =
          remoteOptions != null
              ? remoteOptions.getRemoteDefaultExecProperties()
              : ImmutableSortedMap.of();
    } catch (UserExecException e) {
      throw e.toActionExecutionException(action);
    }

    try {
      actionCacheChecker.updateActionCache(
          action, token, metadataHandler, clientEnv, remoteDefaultProperties);
    } catch (IOException e) {
      // Skyframe has already done all the filesystem access needed for outputs and swallows
      // IOExceptions for inputs. So an IOException is impossible here.
      throw new IllegalStateException(
          "failed to update action cache for " + action.prettyPrint()
              + ", but all outputs should already have been checked", e);
    }
  }

  @Nullable
  List<Artifact> getActionCachedInputs(Action action, PackageRootResolver resolver)
      throws InterruptedException {
    return actionCacheChecker.getCachedInputs(action, resolver);
  }

  /**
   * Perform dependency discovery for action, which must discover its inputs.
   *
   * <p>This method is just a wrapper around {@link Action#discoverInputs} that properly processes
   * any {@link ActionExecutionException} thrown before rethrowing it to the caller.
   */
  NestedSet<Artifact> discoverInputs(
      Action action,
      ActionLookupData actionLookupData,
      MetadataHandler metadataHandler,
      Environment env,
      @Nullable FileSystem actionFileSystem)
      throws ActionExecutionException, InterruptedException {
    FileOutErr fileOutErr =
        actionLogBufferPathGenerator.generate(
            ArtifactPathResolver.createPathResolver(
                actionFileSystem, executorEngine.getExecRoot()));
    ExtendedEventHandler eventHandler = selectEventHandler(action);
    ActionExecutionContext actionExecutionContext =
        ActionExecutionContext.forInputDiscovery(
            executorEngine,
            createFileCache(metadataHandler, actionFileSystem),
            actionInputPrefetcher,
            actionKeyContext,
            metadataHandler,
            env.restartPermitted(),
            lostInputsCheck(actionFileSystem, action, outputService),
            fileOutErr,
            eventHandler,
            clientEnv,
            env,
            actionFileSystem,
            nestedSetExpander);
    if (actionFileSystem != null) {
      updateActionFileSystemContext(
          action,
          actionFileSystem,
          env,
          (output, metadata) -> {
            throw new IllegalStateException(
                "Unexpected output during input discovery: " + output + " (" + metadata + ")");
          },
          /*filesets=*/ ImmutableMap.of());
      // Note that when not using ActionFS, a global setup of the parent directories of the OutErr
      // streams is sufficient.
      setupActionFsFileOutErr(fileOutErr, action);
    }
    eventHandler.post(new ScanningActionEvent(action));

    ActionExecutionException finalException = null;
    try {
      NestedSet<Artifact> artifacts = action.discoverInputs(actionExecutionContext);

      // Input discovery may have been affected by lost inputs. If an action filesystem is used, it
      // may know whether inputs were lost. We should fail fast if any were; rewinding may be able
      // to fix it.
      checkActionFileSystemForLostInputs(actionFileSystem, action, outputService);

      return artifacts;
    } catch (ActionExecutionException e) {
      // Input discovery failures may be caused by lost inputs. Lost input failures have higher
      // priority because rewinding may be able to restore what was lost and allow the action to
      // complete without error.
      if (!(e instanceof LostInputsActionExecutionException)) {
        try {
          checkActionFileSystemForLostInputs(actionFileSystem, action, outputService);
        } catch (LostInputsActionExecutionException lostInputsException) {
          e = lostInputsException;
        }
      }

      if (e instanceof LostInputsActionExecutionException) {
        // If inputs were lost during input discovery, then enrich the exception, informing action
        // rewinding machinery that these lost inputs are now Skyframe deps of the action.
        ((LostInputsActionExecutionException) e).setFromInputDiscovery();
      }
      finalException =
          processAndGetExceptionToThrow(
              env.getListener(),
              actionExecutionContext.getInputPath(action.getPrimaryOutput()),
              action,
              actionLookupData,
              e,
              fileOutErr,
              ErrorTiming.BEFORE_EXECUTION);
      throw finalException;
    } finally {
      eventHandler.post(new StoppedScanningActionEvent(action));
      closeContext(actionExecutionContext, action, finalException);
    }
  }

  boolean isBazelRemoteExecutionEnabled() {
    return bazelRemoteExecutionEnabled;
  }

  private MetadataProvider createFileCache(
      MetadataProvider graphFileCache, @Nullable FileSystem actionFileSystem) {
    if (actionFileSystem instanceof MetadataProvider) {
      return (MetadataProvider) actionFileSystem;
    }
    return new DelegatingPairFileCache(graphFileCache, perBuildFileCache);
  }

  /**
   * This method should be called if the builder encounters an error during execution. This allows
   * the builder to record that it encountered at least one error, and may make it swallow its
   * output to prevent spamming the user any further.
   */
  void recordExecutionError() {
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
    return hadExecutionError && !options.getOptions(KeepGoingOption.class).keepGoing;
  }

  public void configure(
      MetadataProvider fileCache,
      ActionInputPrefetcher actionInputPrefetcher,
      NestedSetExpander nestedSetExpander) {
    this.perBuildFileCache = fileCache;
    this.actionInputPrefetcher = actionInputPrefetcher;
    this.nestedSetExpander = nestedSetExpander;
  }

  /**
   * Temporary interface to allow delegation of action postprocessing to ActionExecutionFunction.
   * The current implementation requires access to local fields in ActionExecutionFunction.
   */
  interface ActionPostprocessing {
    void run(
        Environment env,
        Action action,
        ActionMetadataHandler metadataHandler,
        Map<String, String> clientEnv)
        throws InterruptedException, ActionExecutionException;
  }

  private static ActionContinuationOrResult begin(
      Action action, ActionExecutionContext actionExecutionContext) {
    return new ActionContinuationOrResult() {
      @Override
      public ListenableFuture<?> getFuture() {
        return null;
      }

      @Override
      public ActionContinuationOrResult execute()
          throws ActionExecutionException, InterruptedException {
        return action.beginExecution(actionExecutionContext);
      }
    };
  }

  /** Returns a continuation to run the specified action in a profiler task. */
  private static ActionContinuationOrResult runFully(
      Action action, ActionExecutionContext actionExecutionContext) {
    return new ActionContinuationOrResult() {
      @Override
      public ListenableFuture<?> getFuture() {
        return null;
      }

      @Override
      public ActionContinuationOrResult execute()
          throws ActionExecutionException, InterruptedException {
        return ActionContinuationOrResult.of(action.execute(actionExecutionContext));
      }
    };
  }

  /** Represents an action that needs to be run. */
  private final class ActionRunner extends ActionStep {
    private final Action action;
    private final ActionMetadataHandler metadataHandler;
    private final long actionStartTime;
    private final ActionExecutionContext actionExecutionContext;
    private final ActionLookupData actionLookupData;
    private final ActionExecutionStatusReporter statusReporter;
    private final ActionPostprocessing postprocessing;
    @Nullable private final MetadataAggregator metadataAggregator;

    ActionRunner(
        Action action,
        ActionMetadataHandler metadataHandler,
        long actionStartTime,
        ActionExecutionContext actionExecutionContext,
        ActionLookupData actionLookupData,
        ActionPostprocessing postprocessing,
        @Nullable MetadataAggregator metadataAggregator) {
      this.action = action;
      this.metadataHandler = metadataHandler;
      this.actionStartTime = actionStartTime;
      this.actionExecutionContext = actionExecutionContext;
      this.actionLookupData = actionLookupData;
      this.statusReporter = statusReporterRef.get();
      this.postprocessing = postprocessing;
      this.metadataAggregator = metadataAggregator;
    }

    @SuppressWarnings("LogAndThrow") // Thrown exception shown in user output, not info logs.
    @Override
    public ActionStepOrResult run(Environment env) throws InterruptedException {
      // There are three ExtendedEventHandler instances available while this method is running.
      //   SkyframeActionExecutor.this.reporter
      //   actionExecutionContext.getEventHandler
      //   env.getListener
      // Apparently, one isn't enough.
      //
      // At this time, ProgressLike events that are generated in this class should be posted to
      // env.getListener, while ProgressLike events that are generated in the Action implementation
      // are posted to actionExecutionContext.getEventHandler. The reason for this seems to be
      // action rewinding, which suppresses progress on actionExecutionContext.getEventHandler, for
      // undocumented reasons.
      //
      // It is also unclear why we are posting anything directly to reporter. That probably
      // shouldn't happen.
      try (SilentCloseable c =
          profiler.profileAction(
              ProfilerTask.ACTION,
              action.describe(),
              action.getPrimaryOutput().getExecPathString(),
              getOwnerLabelAsString(action))) {
        String message = action.getProgressMessage();
        if (message != null) {
          reporter.startTask(null, prependExecPhaseStats(message));
        }

        try {
          // It is vital that updateStatus and remove are called in pairs. Unfortunately, if async
          // action execution is enabled, we cannot use a simple finally block, but have to manually
          // ensure that any code path that finishes the state machine also removes the action from
          // the status reporter.
          // To complicate things, the ActionCompletionEvent must _not_ be posted when this action
          // is rewound.
          // TODO(ulfjack): Change the uses of ActionStartedEvent and ActionCompletionEvent such
          // that they can be reposted when rewinding and simplify this code path. Maybe also keep
          // track of the rewind attempt, so that listeners can use that to adjust their behavior.
          ActionStartedEvent event = new ActionStartedEvent(action, actionStartTime);
          statusReporter.updateStatus(event);
          env.getListener().post(event);
          if (!actionFileSystemType().inMemoryFileSystem()) {
            try (SilentCloseable d = profiler.profile(ProfilerTask.INFO, "action.prepare")) {
              // This call generally deletes any files at locations that are declared outputs of the
              // action, although some actions perform additional work, while others intentionally
              // keep previous outputs in place.
              action.prepare(actionExecutionContext.getExecRoot());
            } catch (IOException e) {
              logger.atWarning().withCause(e).log(
                  "failed to delete output files before executing action: '%s'", action);
              throw toActionExecutionException(
                  "failed to delete output files before executing action",
                  e,
                  action,
                  null,
                  Code.ACTION_OUTPUTS_DELETION_FAILURE);
            }
          } else {
            // There's nothing to delete when the action file system is used, but we must ensure
            // that the output directories for stdout and stderr exist.
            setupActionFsFileOutErr(actionExecutionContext.getFileOutErr(), action);
          }
          if (actionFileSystemType().inMemoryFileSystem()) {
            createActionFsOutputDirectories(action, actionExecutionContext);
          } else {
            createOutputDirectories(action, actionExecutionContext);
          }
        } catch (ActionExecutionException e) {
          // This try-catch block cannot trigger rewinding, so it is safe to notify the status
          // reporter and also post the ActionCompletionEvent.
          notifyActionCompletion(env.getListener(), /*postActionCompletionEvent=*/ true);
          return ActionStepOrResult.of(e);
        }

        // This is the first iteration of the async action execution framework. It is currently only
        // implemented for SpawnAction (and subclasses), and will need to be extended for all other
        // action types.
        if (useAsyncExecution) {
          // TODO(ulfjack): This causes problems in that REMOTE_EXECUTION segments now heavily
          // overlap in the Json profile, which the renderer isn't able to handle. We should move
          // those to some sort of 'virtual thread' to visualize the work that's happening on other
          // machines.
          return continueAction(
              actionExecutionContext.getEventHandler(), begin(action, actionExecutionContext));
        }

        return continueAction(env.getListener(), runFully(action, actionExecutionContext));
      }
    }

    private String getOwnerLabelAsString(Action action) {
      ActionOwner owner = action.getOwner();
      if (owner == null) {
        return "";
      }
      Label ownerLabel = owner.getLabel();
      if (ownerLabel == null) {
        return "";
      }
      return ownerLabel.getCanonicalForm();
    }

    private void notifyActionCompletion(
        ExtendedEventHandler eventHandler, boolean postActionCompletionEvent) {
      statusReporter.remove(action);
      if (postActionCompletionEvent) {
        eventHandler.post(new ActionCompletionEvent(actionStartTime, action, actionLookupData));
      }
      String message = action.getProgressMessage();
      if (message != null) {
        // Tell the receiver that the action has completed *before* telling the reporter.
        // This way the latter will correctly show the number of completed actions when task
        // completion messages are enabled (--show_task_finish).
        completionReceiver.actionCompleted(actionLookupData);
        reporter.finishTask(null, prependExecPhaseStats(message));
      }
    }

    /** Executes the given continuation and returns a new one or a final result. */
    private ActionStepOrResult continueAction(
        ExtendedEventHandler eventHandler, ActionContinuationOrResult actionContinuation)
        throws InterruptedException {
      // Every code path that exits this method must call notifyActionCompletion, except for the
      // one that returns a new ActionContinuationStep. Unfortunately, that requires some code
      // duplication.
      ActionContinuationOrResult nextActionContinuationOrResult;
      try (SilentCloseable c = profiler.profile(ProfilerTask.INFO, "ActionContinuation.execute")) {
        nextActionContinuationOrResult = actionContinuation.execute();

        // An action's result (or intermediate state) may have been affected by lost inputs. If an
        // action filesystem is used, it may know whether inputs were lost. We should fail fast if
        // any were; rewinding may be able to fix it.
        checkActionFileSystemForLostInputs(
            actionExecutionContext.getActionFileSystem(), action, outputService);
      } catch (ActionExecutionException e) {

        // Action failures may be caused by lost inputs. Lost input failures have higher priority
        // because rewinding may be able to restore what was lost and allow the action to complete
        // without error.
        if (!(e instanceof LostInputsActionExecutionException)) {
          try {
            checkActionFileSystemForLostInputs(
                actionExecutionContext.getActionFileSystem(), action, outputService);
          } catch (LostInputsActionExecutionException lostInputsException) {
            e = lostInputsException;
          }
        }

        boolean isLostInputsException = e instanceof LostInputsActionExecutionException;
        if (isLostInputsException) {
          ((LostInputsActionExecutionException) e).setActionStartedEventAlreadyEmitted();
        }
        notifyActionCompletion(eventHandler, /*postActionCompletionEvent=*/ !isLostInputsException);
        return ActionStepOrResult.of(
            processAndGetExceptionToThrow(
                eventHandler,
                actionExecutionContext.getInputPath(action.getPrimaryOutput()),
                action,
                actionLookupData,
                e,
                actionExecutionContext.getFileOutErr(),
                ErrorTiming.AFTER_EXECUTION));
      } catch (InterruptedException e) {
        notifyActionCompletion(eventHandler, /*postActionCompletionEvent=*/ true);
        return ActionStepOrResult.of(e);
      }

      if (!nextActionContinuationOrResult.isDone()) {
        return new ActionContinuationStep(nextActionContinuationOrResult);
      }

      try {
        ActionExecutionValue actionExecutionValue;
        try (SilentCloseable c =
            profiler.profile(ProfilerTask.ACTION_COMPLETE, "actuallyCompleteAction")) {
          actionExecutionValue =
              actuallyCompleteAction(eventHandler, nextActionContinuationOrResult.get());
        }
        return new ActionCacheWriteStep(actionExecutionValue);
      } catch (ActionExecutionException e) {
        return ActionStepOrResult.of(e);
      } finally {
        notifyActionCompletion(eventHandler, /*postActionCompletionEvent=*/ true);
      }
    }

    @SuppressWarnings("LogAndThrow") // Thrown exception shown in user output, not info logs.
    private ActionExecutionValue actuallyCompleteAction(
        ExtendedEventHandler eventHandler, ActionResult actionResult)
        throws ActionExecutionException, InterruptedException {
      boolean outputAlreadyDumped = false;
      if (actionResult != ActionResult.EMPTY) {
        eventHandler.post(new ActionResultReceivedEvent(action, actionResult));
      }

      // Action terminated fine, now report the output.
      // The .showOutput() method is not necessarily a quick check: in its
      // current implementation it uses regular expression matching.
      FileOutErr outErrBuffer = actionExecutionContext.getFileOutErr();
      if (outErrBuffer.hasRecordedOutput()) {
        if (replayActionOutErr) {
          dumpRecordedOutErr(actionExecutionContext.getEventHandler(), action, outErrBuffer);
          outputAlreadyDumped = true;
        } else if (action.showsOutputUnconditionally()
            || reporter.showOutput(Label.print(action.getOwner().getLabel()))) {
          dumpRecordedOutErr(reporter, action, outErrBuffer);
          outputAlreadyDumped = true;
        }
      }

      MetadataHandler metadataHandler = actionExecutionContext.getMetadataHandler();
      FileOutErr fileOutErr = actionExecutionContext.getFileOutErr();
      Path primaryOutputPath = actionExecutionContext.getInputPath(action.getPrimaryOutput());
      try {
        Preconditions.checkState(action.inputsDiscovered(),
            "Action %s successfully executed, but inputs still not known", action);

        if (metadataAggregator != null) {
          metadataAggregator.finish();
        }

        if (!checkOutputs(action, metadataHandler)) {
          throw toActionExecutionException(
              "not all outputs were created or valid",
              null,
              action,
              outputAlreadyDumped ? null : fileOutErr,
              Code.ACTION_OUTPUTS_NOT_CREATED);
        }

        if (outputService != null && finalizeActions) {
          try (SilentCloseable c =
              profiler.profile(ProfilerTask.INFO, "outputService.finalizeAction")) {
            outputService.finalizeAction(action, metadataHandler);
          } catch (EnvironmentalExecException | IOException e) {
            logger.atWarning().withCause(e).log("unable to finalize action: '%s'", action);
            throw toActionExecutionException(
                "unable to finalize action",
                e,
                action,
                fileOutErr,
                Code.ACTION_FINALIZATION_FAILURE);
          }
        }

        reportActionExecution(
            eventHandler,
            primaryOutputPath,
            action,
            actionResult,
            actionFileSystemType().inMemoryFileSystem(),
            null,
            fileOutErr,
            ErrorTiming.NO_ERROR);
      } catch (ActionExecutionException actionException) {
        // Success in execution but failure in completion.
        reportActionExecution(
            eventHandler,
            primaryOutputPath,
            action,
            actionResult,
            actionFileSystemType().inMemoryFileSystem(),
            actionException,
            fileOutErr,
            ErrorTiming.AFTER_EXECUTION);
        throw actionException;
      } catch (IllegalStateException exception) {
        // More serious internal error, but failure still reported.
        reportActionExecution(
            eventHandler,
            primaryOutputPath,
            action,
            actionResult,
            actionFileSystemType().inMemoryFileSystem(),
            new ActionExecutionException(
                exception,
                action,
                true,
                CrashFailureDetails.detailedExitCodeForThrowable(exception)),
            fileOutErr,
            ErrorTiming.AFTER_EXECUTION);
        throw exception;
      }

      Preconditions.checkState(
          actionExecutionContext.getOutputSymlinks() == null
              || action instanceof SkyframeAwareAction,
          "Unexpected to find outputSymlinks set"
              + " in an action which is not a SkyframeAwareAction. Action: %s\n symlinks:%s",
          action,
          actionExecutionContext.getOutputSymlinks());
      return ActionExecutionValue.createFromOutputStore(
          this.metadataHandler.getOutputStore(),
          actionExecutionContext.getOutputSymlinks(),
          (action instanceof IncludeScannable)
              ? ((IncludeScannable) action).getDiscoveredModules()
              : null,
          actionDependsOnBuildId(action));
    }

    /** A closure to continue an asynchronously running action. */
    private class ActionContinuationStep extends ActionStep {
      private final ActionContinuationOrResult actionContinuationOrResult;

      public ActionContinuationStep(ActionContinuationOrResult actionContinuationOrResult) {
        Preconditions.checkArgument(!actionContinuationOrResult.isDone());
        this.actionContinuationOrResult = actionContinuationOrResult;
      }

      @Override
      public ActionStepOrResult run(Environment env) throws InterruptedException {
        ListenableFuture<?> future = actionContinuationOrResult.getFuture();
        if (future != null && !future.isDone()) {
          env.dependOnFuture(future);
          return this;
        }
        return continueAction(actionExecutionContext.getEventHandler(), actionContinuationOrResult);
      }
    }

    /** A closure to post-process the action and write the result to the action cache. */
    private class ActionCacheWriteStep extends ActionStep {
      private final ActionExecutionValue value;

      public ActionCacheWriteStep(ActionExecutionValue value) {
        this.value = value;
      }

      @Override
      public ActionStepOrResult run(Environment env) {
        try (SilentCloseable c = profiler.profile(ProfilerTask.INFO, "postprocessing.run")) {
          postprocessing.run(env, action, metadataHandler, actionExecutionContext.getClientEnv());
          if (env.valuesMissing()) {
            return this;
          }
        } catch (InterruptedException e) {
          return ActionStepOrResult.of(e);
        } catch (ActionExecutionException e) {
          return ActionStepOrResult.of(e);
        }
        return ActionStepOrResult.of(value);
      }
    }
  }

  /**
   * Create output directories for an ActionFS. The action-local filesystem starts empty, so we
   * expect the output directory creation to always succeed. There can be no interference from state
   * left behind by prior builds or other actions intra-build.
   */
  private void createActionFsOutputDirectories(Action action, ActionExecutionContext context)
      throws ActionExecutionException {
    try {
      Set<Path> done = new HashSet<>(); // avoid redundant calls for the same directory.
      for (Artifact outputFile : action.getOutputs()) {
        Path outputDir;
        if (outputFile.isTreeArtifact()) {
          outputDir = context.getPathResolver().toPath(outputFile);
        } else {
          outputDir = context.getPathResolver().toPath(outputFile).getParentDirectory();
        }

        if (done.add(outputDir)) {
          outputDir.createDirectoryAndParents();
        }
      }
    } catch (IOException e) {
      String message = "failed to create output directory: " + e.getMessage();
      DetailedExitCode code =
          createDetailedExitCode(message, Code.ACTION_FS_OUTPUT_DIRECTORY_CREATION_FAILURE);
      ActionExecutionException ex = new ActionExecutionException(message, e, action, false, code);
      printError(ex.getMessage(), action, null);
      throw ex;
    }
  }

  /**
   * Ensure that no symlinks exists between the output root and the output file. These are all
   * expected to be regular directories. Violations of this expectations can only come from state
   * left behind by previous invocations or external filesystem mutation.
   */
  private void symlinkCheck(
      final Path dir, final Artifact outputFile, ActionExecutionContext context)
      throws IOException {
    PathFragment root = outputFile.getRoot().getRoot().asPath().asFragment();
    Path curDir = context.getPathResolver().convertPath(dir);
    Set<PathFragment> checkDirs = new HashSet<>();
    while (!curDir.asFragment().equals(root)) {
      // Fast path: Somebody already checked that this is a regular directory this invocation.
      if (knownRegularDirectories.contains(curDir.asFragment())) {
        return;
      }
      if (!curDir.isDirectory(Symlinks.NOFOLLOW)) {
        throw new IOException(curDir + " is not a regular directory");
      }
      checkDirs.add(curDir.asFragment());
      curDir = curDir.getParentDirectory();
    }

    // Defer adding to known regular directories until we've checked all parent directories.
    knownRegularDirectories.addAll(checkDirs);
  }

  private void createOutputDirectories(Action action, ActionExecutionContext context)
      throws ActionExecutionException {
    try {
      Set<Path> done = new HashSet<>(); // avoid redundant calls for the same directory.
      for (Artifact outputFile : action.getOutputs()) {
        Path outputDir;
        if (outputFile.isTreeArtifact()) {
          outputDir = context.getPathResolver().toPath(outputFile);
        } else {
          outputDir = context.getPathResolver().toPath(outputFile).getParentDirectory();
        }

        if (done.add(outputDir)) {
          try {
            if (!knownRegularDirectories.contains(outputDir.asFragment())) {
              outputDir.createDirectoryAndParents();
              symlinkCheck(outputDir, outputFile, context);
            }
            continue;
          } catch (IOException e) {
            /* Fall through to plan B. */
          }

          // Possibly some direct ancestors are not directories.  In that case, we traverse the
          // ancestors downward, deleting any non-directories. This handles the case where a file
          // becomes a directory. The traversal is done downward because otherwise we may delete
          // files through a symlink in a parent directory. Since Blaze never creates such
          // directories within a build, we have no idea where on disk we're actually deleting.
          //
          // Symlinks should not be followed so in order to clean up symlinks pointing to Fileset
          // outputs from previous builds. See bug [incremental build of Fileset fails if
          // Fileset.out was changed to be a subdirectory of the old value].
          try {
            Path p =
                context.getPathResolver().transformRoot(outputFile.getRoot().getRoot()).asPath();
            PathFragment relativePath = outputDir.relativeTo(p);
            for (int i = 0; i < relativePath.segmentCount(); i++) {
              p = p.getRelative(relativePath.getSegment(i));

              // This lock ensures that the only thread that observes a filesystem transition in
              // which the path p first exists and then does not is the thread that calls
              // p.delete() and causes the transition.
              //
              // If it were otherwise, then some thread A could test p.exists(), see that it does,
              // then test p.isDirectory(), see that p isn't a directory (because, say, thread
              // B deleted it), and then call p.delete(). That could result in two different kinds
              // of failures:
              //
              // 1) In the time between when thread A sees that p is not a directory and when thread
              // A calls p.delete(), thread B may reach the call to createDirectoryAndParents
              // and create a directory at p, which thread A then deletes. Thread B would then try
              // adding outputs to the directory it thought was there, and fail.
              //
              // 2) In the time between when thread A sees that p is not a directory and when thread
              // A calls p.delete(), thread B may create a directory at p, and then either create a
              // subdirectory beneath it or add outputs to it. Then when thread A tries to delete p,
              // it would fail.
              Lock lock = outputDirectoryDeletionLock.get(p);
              lock.lock();
              try {
                FileStatus stat = p.statIfFound(Symlinks.NOFOLLOW);
                if (stat == null) {
                  // Missing entry: Break out and create expected directories.
                  break;
                }
                if (stat.isDirectory()) {
                  // If this directory used to be a tree artifact it won't be writable.
                  p.setWritable(true);
                  knownRegularDirectories.add(p.asFragment());
                } else {
                  // p may be a file or symlink (possibly from a Fileset in a previous build).
                  p.delete(); // throws IOException
                  break;
                }
              } finally {
                lock.unlock();
              }
            }
            outputDir.createDirectoryAndParents();
          } catch (IOException e) {
            String message =
                String.format(
                    "failed to create output directory '%s': %s", outputDir, e.getMessage());
            DetailedExitCode code =
                createDetailedExitCode(message, Code.ACTION_OUTPUT_DIRECTORY_CREATION_FAILURE);
            throw new ActionExecutionException(message, e, action, false, code);
          }
        }
      }
    } catch (ActionExecutionException ex) {
      printError(ex.getMessage(), action, null);
      throw ex;
    }
  }

  private String prependExecPhaseStats(String message) {
    // Prints a progress message like:
    //   [2608/6445] Compiling foo/bar.cc [host]
    return progressSupplier.getProgressString() + " " + message;
  }

  private static void setupActionFsFileOutErr(FileOutErr fileOutErr, Action action)
      throws ActionExecutionException {
    try {
      fileOutErr.getOutputPath().getParentDirectory().createDirectoryAndParents();
      fileOutErr.getErrorPath().getParentDirectory().createDirectoryAndParents();
    } catch (IOException e) {
      String message =
          String.format(
              "failed to create output directory for output streams'%s': %s",
              fileOutErr.getErrorPath(), e.getMessage());
      DetailedExitCode code =
          createDetailedExitCode(message, Code.ACTION_FS_OUT_ERR_DIRECTORY_CREATION_FAILURE);
      throw new ActionExecutionException(message, e, action, false, code);
    }
  }

  ActionExecutionException processAndGetExceptionToThrow(
      ExtendedEventHandler eventHandler,
      Path primaryOutputPath,
      Action action,
      ActionLookupData actionLookupData,
      ActionExecutionException e,
      FileOutErr outErrBuffer,
      ErrorTiming errorTiming) {
    if (e instanceof LostInputsActionExecutionException) {
      // If inputs are lost, then avoid publishing ActionExecutedEvent or reporting the error.
      // Action rewinding will rerun this failed action after trying to regenerate the lost inputs.
      // Enrich the exception so it can be distinguished by shared actions getting cache hits and so
      // that, if rewinding fails, an ActionExecutedEvent can be published, and the error reported.
      LostInputsActionExecutionException lostInputsException =
          (LostInputsActionExecutionException) e;
      lostInputsException.setPrimaryAction(actionLookupData);
      lostInputsException.setPrimaryOutputPath(primaryOutputPath);
      lostInputsException.setFileOutErr(outErrBuffer);
      return lostInputsException;
    }

    reportActionExecution(
        eventHandler,
        primaryOutputPath,
        action,
        null,
        actionFileSystemType().inMemoryFileSystem(),
        e,
        outErrBuffer,
        errorTiming);
    boolean reported = reportErrorIfNotAbortingMode(e, outErrBuffer);

    ActionExecutionException toThrow = e;
    if (reported){
      // If we already printed the error for the exception we mark it as already reported
      // so that we do not print it again in upper levels.
      // Note that we need to report it here since we want immediate feedback of the errors
      // and in some cases the upper-level printing mechanism only prints one of the errors.
      toThrow = new AlreadyReportedActionExecutionException(e);
    }

    // Now, return the exception to rethrow. This can have two effects:
    // If we're still building, the exception will get retrieved by the completor and rethrown.
    // If we're aborting, the exception will never be retrieved from the completor, since the
    // completor is waiting for all outstanding jobs to finish. After they have finished, it will
    // only rethrow the exception that initially caused it to abort will and not check the exit
    // status of any actions that had finished in the meantime.
    return toThrow;
  }

  private static void reportMissingOutputFile(
      Action action, Artifact output, Reporter reporter, boolean isSymlink, IOException exception) {
    boolean genrule = action.getMnemonic().equals("Genrule");
    String prefix = (genrule ? "declared output '" : "output '") + output.prettyPrint() + "' ";
    logger.atWarning().log(
        "Error creating %s%s%s: %s",
        isSymlink ? "symlink " : "", prefix, genrule ? " by genrule" : "", exception.getMessage());
    if (isSymlink) {
      String msg = prefix + "is a dangling symbolic link";
      reporter.handle(Event.error(action.getOwner().getLocation(), msg));
    } else {
      String suffix = genrule ? " by genrule. This is probably "
          + "because the genrule actually didn't create this output, or because the output was a "
          + "directory and the genrule was run remotely (note that only the contents of "
          + "declared file outputs are copied from genrules run remotely)" : "";
      reporter.handle(Event.error(
          action.getOwner().getLocation(), prefix + "was not created" + suffix));
    }
  }

  private static void reportOutputTreeArtifactErrors(
      Action action, Artifact output, Reporter reporter, IOException e) {
    String errorMessage;
    if (e instanceof FileNotFoundException) {
      errorMessage = String.format("TreeArtifact %s was not created", output.prettyPrint());
    } else {
      errorMessage = String.format(
          "Error while validating output TreeArtifact %s : %s", output, e.getMessage());
    }

    reporter.handle(Event.error(action.getOwner().getLocation(), errorMessage));
  }

  /**
   * Validates that all action input contents were not lost if they were read, and if an action file
   * system was used. Throws a {@link LostInputsActionExecutionException} describing the lost inputs
   * if any were.
   */
  private static void checkActionFileSystemForLostInputs(
      @Nullable FileSystem actionFileSystem, Action action, OutputService outputService)
      throws LostInputsActionExecutionException {
    if (actionFileSystem != null) {
      outputService.checkActionFileSystemForLostInputs(actionFileSystem, action);
    }
  }

  private static LostInputsCheck lostInputsCheck(
      @Nullable FileSystem actionFileSystem, Action action, OutputService outputService) {
    return actionFileSystem == null
        ? LostInputsCheck.NONE
        : () -> outputService.checkActionFileSystemForLostInputs(actionFileSystem, action);
  }

  /**
   * Validates that all action outputs were created or intentionally omitted. This can result in
   * chmod calls on the output files; see {@link ActionMetadataHandler}.
   *
   * @return false if some outputs are missing, true - otherwise.
   */
  private boolean checkOutputs(Action action, MetadataHandler metadataHandler) {
    boolean success = true;
    for (Artifact output : action.getOutputs()) {
      // getMetadata has the side effect of adding the artifact to the cache if it's not there
      // already (e.g., due to a previous call to MetadataHandler.injectDigest), therefore we only
      // call it if we know the artifact is not omitted.
      if (!metadataHandler.artifactOmitted(output)) {
        try {
          metadataHandler.getMetadata(output);
        } catch (IOException e) {
          success = false;
          if (output.isTreeArtifact()) {
            reportOutputTreeArtifactErrors(action, output, reporter, e);
          } else {
            // Are all exceptions caught due to missing files?
            reportMissingOutputFile(action, output, reporter, output.getPath().isSymbolicLink(), e);
          }
        }
      }
    }
    return success;
  }

  /**
   * Convenience function for creating an ActionExecutionException reporting that the action failed
   * due to a the exception cause, if there is an additional explanatory message that clarifies the
   * message of the exception. Combines the user-provided message and the exceptions' message and
   * reports the combination as error.
   *
   * @param message A small text that explains why the action failed
   * @param cause The exception that caused the action to fail
   * @param action The action that failed
   * @param actionOutput The output of the failed Action. May be null, if there is no output to
   *     display
   * @param detailedCode The fine-grained failure code describing the failure
   */
  private ActionExecutionException toActionExecutionException(
      String message,
      Throwable cause,
      Action action,
      FileOutErr actionOutput,
      FailureDetails.Execution.Code detailedCode) {
    DetailedExitCode code = createDetailedExitCode(message, detailedCode);
    ActionExecutionException ex;
    if (cause == null) {
      ex = new ActionExecutionException(message, action, false, code);
    } else {
      ex = new ActionExecutionException(message, cause, action, false, code);
    }
    printError(ex.getMessage(), action, actionOutput);
    return ex;
  }

  private static DetailedExitCode createDetailedExitCode(String message, Code detailedCode) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setExecution(Execution.newBuilder().setCode(detailedCode))
            .build());
  }

  /**
   * For the action 'action' that failed due to 'message' with the output 'actionOutput', notify the
   * user about the error. To notify the user, the method first displays the output of the action
   * and then reports an error via the reporter. The method ensures that the two messages appear
   * next to each other by locking the outErr object where the output is displayed.
   *
   * <p>Should not be called for actions that might have failed because the build is shutting down
   * after an error, since it prints output unconditionally, and such output should be suppressed.
   *
   * @param message The reason why the action failed
   * @param action The action that failed, must not be null.
   * @param actionOutput The output of the failed Action. May be null, if there is no output to
   *     display
   */
  @SuppressWarnings("SynchronizeOnNonFinalField")
  void printError(String message, ActionAnalysisMetadata action, FileOutErr actionOutput) {
    synchronized (reporter) {
      if (options.getOptions(KeepGoingOption.class).keepGoing) {
        message = "Couldn't " + describeAction(action) + ": " + message;
      }
      Event event = Event.error(action.getOwner().getLocation(), message);
      dumpRecordedOutErr(reporter, event, actionOutput);
      recordExecutionError();
    }
  }

  /** Describe an action, for use in error messages. */
  private static String describeAction(ActionAnalysisMetadata action) {
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
  private void dumpRecordedOutErr(
      EventHandler eventHandler, Action action, FileOutErr outErrBuffer) {
    Event event =
        replayActionOutErr
            // Info events are not cached in Skyframe, so we make this a warning.
            ? Event.warn("From " + action.describe() + ":")
            : Event.info("From " + action.describe() + ":");
    dumpRecordedOutErr(eventHandler, event, outErrBuffer);
  }

  /**
   * Dump the output from the action.
   *
   * @param prefixEvent An event to post before dumping the output
   * @param outErrBuffer The OutErr that recorded the actions output
   */
  private void dumpRecordedOutErr(
      EventHandler eventHandler, Event prefixEvent, FileOutErr outErrBuffer) {
    // Only print the output if we're not winding down.
    if (isBuilderAborting()) {
      return;
    }
    if (outErrBuffer != null && outErrBuffer.hasRecordedOutput()) {
      // Bind the output to the prefix event.
      eventHandler.handle(prefixEvent.withStdoutStderr(outErrBuffer));
    } else {
      eventHandler.handle(prefixEvent);
    }
  }

  private static void reportActionExecution(
      ExtendedEventHandler eventHandler,
      Path primaryOutputPath,
      Action action,
      @Nullable ActionResult actionResult,
      boolean isInMemoryFs,
      ActionExecutionException exception,
      FileOutErr outErr,
      ErrorTiming errorTiming) {
    Path stdout = null;
    Path stderr = null;
    ImmutableList<MetadataLog> logs = ImmutableList.of();

    if (outErr.hasRecordedStdout()) {
      stdout = outErr.getOutputPath();
    }
    if (outErr.hasRecordedStderr()) {
      stderr = outErr.getErrorPath();
    }
    if (actionResult != null) {
      logs =
          actionResult.spawnResults().stream()
              .filter(spawnResult -> spawnResult.getActionMetadataLog().isPresent())
              .map(spawnResult -> spawnResult.getActionMetadataLog().get())
              .collect(ImmutableList.toImmutableList());
    }
    eventHandler.post(
        new ActionExecutedEvent(
            action.getPrimaryOutput().getExecPath(),
            action,
            exception,
            primaryOutputPath,
            stdout,
            stderr,
            logs,
            errorTiming,
            isInMemoryFs));
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
  @SuppressWarnings("SynchronizeOnNonFinalField")
  private boolean reportErrorIfNotAbortingMode(
      ActionExecutionException ex, FileOutErr outErrBuffer) {
    // For some actions (e.g., many local actions) the pollInterruptedStatus()
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
    void actionCompleted(ActionLookupData actionLookupData);
    /** Notes that an action has started, giving the key. */
    void noteActionEvaluationStarted(ActionLookupData actionLookupData, Action action);
  }

  public void setActionExecutionProgressReportingObjects(
      @Nullable ProgressSupplier progressSupplier,
      @Nullable ActionCompletedReceiver completionReceiver) {
    this.progressSupplier = progressSupplier;
    this.completionReceiver = completionReceiver;
  }

  private static final class DelegatingPairFileCache implements MetadataProvider {
    private final MetadataProvider perActionCache;
    private final MetadataProvider perBuildFileCache;

    private DelegatingPairFileCache(
        MetadataProvider mainCache, MetadataProvider perBuildFileCache) {
      this.perActionCache = mainCache;
      this.perBuildFileCache = perBuildFileCache;
    }

    @Override
    public FileArtifactValue getMetadata(ActionInput input) throws IOException {
      FileArtifactValue metadata = perActionCache.getMetadata(input);
      return (metadata != null) && (metadata != FileArtifactValue.MISSING_FILE_MARKER)
          ? metadata
          : perBuildFileCache.getMetadata(input);
    }

    @Override
    public ActionInput getInput(String execPath) {
      ActionInput input = perActionCache.getInput(execPath);
      return input != null ? input : perBuildFileCache.getInput(execPath);
    }
  }

  /**
   * Assists with aggregation of tree artifacts for an action file system which is only aware of
   * individual outputs.
   */
  private static final class MetadataAggregator implements MetadataConsumer {
    private final TreeArtifactValue.MultiBuilder treeArtifacts =
        TreeArtifactValue.newConcurrentMultiBuilder();
    private final MetadataInjector metadataInjector;

    MetadataAggregator(MetadataInjector metadataInjector) {
      this.metadataInjector = metadataInjector;
    }

    @Override
    public void accept(Artifact output, FileArtifactValue metadata) {
      if (output.isChildOfDeclaredDirectory()) {
        treeArtifacts.putChild((TreeFileArtifact) output, metadata);
      } else {
        metadataInjector.injectFile(output, metadata);
      }
    }

    void finish() {
      treeArtifacts.injectTo(metadataInjector);
    }
  }
}
