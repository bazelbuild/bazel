// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action.MiddlemanType;
import com.google.devtools.build.lib.actions.ActionCacheChecker.Token;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An abstract Builder implementation that takes care of the building of
 * prerequisites, iteration over artifacts, and other common features.
 */
@ConditionallyThreadCompatible // Condition: at most one builder per workspace.
abstract class AbstractBuilder implements Builder {

  private static Logger LOG = Logger.getLogger(AbstractBuilder.class.getName());
  private final ActionInputFileCache fileCache;

  // Cache the log-level in these booleans to avoid unnecessary calls to
  // logging code on critical path.
  protected boolean LOG_FINE;
  protected boolean LOG_FINER;
  protected boolean LOG_FINEST;

  protected final boolean keepGoing;

  protected final Reporter reporter;

  /**
   * A flag that starts false and becomes true if an action fails.
   */
  private boolean hadExecutionError;
  private boolean explain;

  protected final DependencyChecker dependencyChecker;

  protected final Stack<Artifact> fileStack = new Stack<>();

  // A map from Actions to error messages.  The keys are actions that we've
  // discovered are "bad" for some reason and should be failed during
  // execution; see checkActionGraph().  (Technically a concurrent map isn't
  // needed since we only modify it prior to the concurrent phase.)
  protected Map<Action, String> badActions = null;

  private final Profiler profiler = Profiler.instance();

  protected Executor executor;

  /**
   * The total estimated workload for building the transitive closure of the top level
   * artifacts passed to buildArtifacts.  The total is set by initBuild().
   */
  private long workloadTotal;

  /**
   * The amount of estimated work completed thus far in the build.
   */
  protected AtomicLong workCompleted;

  /**
   * The amount of work for which we got a hit in the action cache.
   * This is the amount of estimated work completed thus far in the build
   * for which we did not actually do any work other than dependency
   * checking, because the action's output was cached in the action cache
   * and the action and its inputs hadn't changed.
   */
  private AtomicLong workSavedByActionCache;

  protected final EventBus eventBus;

  protected DependentActionGraph forwardGraph;
  protected final ActionExecutor actionExecutor;

  private final ActionLogBufferPathGenerator actionLogBufferPathGenerator;

  protected AbstractBuilder(Reporter reporter, EventBus eventBus,
      DependencyChecker dependencyChecker,
      boolean keepGoing,
      Path outputRoot, ActionInputFileCache fileCache) {
    this.fileCache = fileCache;
    this.reporter = Preconditions.checkNotNull(reporter);
    this.eventBus = Preconditions.checkNotNull(eventBus);
    this.keepGoing = keepGoing;
    this.dependencyChecker = Preconditions.checkNotNull(dependencyChecker);
    this.actionLogBufferPathGenerator = new ActionLogBufferPathGenerator(outputRoot);
    this.actionExecutor = new ActionExecutor(reporter);
    actionExecutor.setActionLogBufferPathGenerator(actionLogBufferPathGenerator);
  }

  protected class ActionExecutor extends AbstractActionExecutor {

    public ActionExecutor(Reporter reporter) {
      super(reporter);
    }

    @Override
    protected void postEvent(Object event) {
      eventBus.post(event);
    }

    @Override
    protected void printError(String message, Action action, FileOutErr actionOutput) {
      synchronized (reporter) {
        if (actionOutput != null && actionOutput.hasRecordedOutput()) {
          dumpRecordedOutErr(action, actionOutput);
        }
        if (keepGoing) {
          message = "Couldn't " + describeAction(action) + ": " + message;
        }
        reporter.handle(Event.error(action.getOwner().getLocation(), message));
        recordExecutionError();
      }
    }

    @Override
    protected void updateCache(Action action, Token token, MetadataHandler metadataHandler)
        throws ActionExecutionException {
      try {
        dependencyChecker.afterExecution(action, token);
      } catch (IOException e) {
        this.reportError("failed to update action cache", e, action);
      }
    }

    @Override
    protected boolean isBuilderAborting() {
      return AbstractBuilder.this.isBuilderAborting();
    }

    @Override
    protected void scheduleAndExecuteAction(Action action,
        ActionExecutionContext actionExecutionContext)
    throws ActionExecutionException, InterruptedException {
      AbstractBuilder.this.scheduleAndExecuteAction(action, actionExecutionContext);
    }

    @Override
    protected boolean reportErrorIfNotAbortingMode(ActionExecutionException ex,
        FileOutErr outErrBuffer) {
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
  }

  //---------------------------------------------------------------------------
  //
  // Subclass hooks
  //

  /**
   * Clear Builder's data structures. Usually this means clearing
   * memory of actions it has already visited and other related state.
   *
   * Requesting that the Builder do the equivalent of "make A B" does not cause
   * it re-visit (traverse) all the common parts of A and B twice, due to the
   * Builder's memory of visited Actions.
   */
  protected void clear() {
    badActions = null;
  }

  /**
   * To be called before initiating a build.
   */
  private void initBuild(Set<Artifact> artifactSet, DependentActionGraph forwardGraph,
      ModifiedFileSet modifiedFileSet, Set<Artifact> builtArtifacts, boolean explain)
      throws InterruptedException {
    this.forwardGraph = forwardGraph;
    clear();
    Preconditions.checkState(builtArtifacts == null || builtArtifacts.isEmpty());

    if (!fileStack.isEmpty()) {
      throw new AssertionError(fileStack.toString());
    }
    LOG_FINE = LOG.isLoggable(Level.FINE);
    LOG_FINER = LOG.isLoggable(Level.FINER);
    LOG_FINEST = LOG.isLoggable(Level.FINEST);

    this.explain = explain;
    // estimate the work to build the actions
    this.workloadTotal = forwardGraph.estimateTotalWorkload();
    this.workCompleted = new AtomicLong(0);
    this.workSavedByActionCache = new AtomicLong(0);

    badActions = forwardGraph.checkActionGraph(artifactSet);
    dependencyChecker.init(artifactSet, builtArtifacts, forwardGraph,
        executor, modifiedFileSet, reporter);
  }

  /**
   * Build the specified files, and all necessary prerequisites.
   *
   * <p>{@link AbstractBuilder#buildArtifacts} ensures that necessary initializations have been
   * performed, see {@link AbstractBuilder#initBuild}.
   *
   * @throws TestExecException if any test fails
   */
  protected abstract void buildArtifactsHook(Set<Artifact> artifactSet,
      DependentActionGraph forwardGraph, ModifiedFileSet modified, Set<Artifact> builtArtifacts)
      throws BuildFailedException, InterruptedException, TestExecException;

  @Override
  public final void buildArtifacts(Set<Artifact> artifactSet,
      Set<Artifact> exclusiveTestArtifacts,
      DependentActionGraph forwardGraph,
      Executor executor,
      ModifiedFileSet modified,
      Set<Artifact> builtArtifacts, boolean explain)
          throws BuildFailedException, InterruptedException, TestExecException {
    Preconditions.checkState(artifactSet.size() ==
                             forwardGraph.getTopLevelAction().getUnbuiltInputs(),
                             "Set: %s, action: %s", artifactSet, forwardGraph.getTopLevelAction());
    Preconditions.checkState(exclusiveTestArtifacts.isEmpty());
    this.executor = executor;
    actionExecutor.setExecutorEngine(executor);
    initBuild(artifactSet, forwardGraph, modified, builtArtifacts, explain);  // calls clear()
    buildArtifactsHook(artifactSet, forwardGraph, modified, builtArtifacts);
  }

  //---------------------------------------------------------------------------
  //
  // Protected methods: subroutines for use by derived classes.
  //

  protected final Profiler getProfiler() {
    return profiler;
  }

  /**
   * This method should be called if the builder encounters an error during
   * execution. This allows the builder to record that it encountered at
   * least one error, and may make it swallow its' output to prevent
   * spamming the user any further.
   */
  protected void recordExecutionError() {
    hadExecutionError = true;
  }

  /**
   * Returns true if the Builder is winding down (i.e. cancelling outstanding
   * actions and preparing to abort.)
   * The builder is winding down iff:
   * - we had an execution error
   * - we are not running with --keep_going
   */
  protected boolean isBuilderAborting() {
    return hadExecutionError && !keepGoing;
  }

  /**
   * Check the dependencies of the specified action, and execute it if
   * required.  Also, after the action has been executed,
   * check that the outputs exist and make them read-only.
   *
   * This is thread-safe so long as you don't try to execute the same action
   * twice at the same time (or overlapping times).
   * May execute in a worker thread.
   *
   * @throws ActionExecutionException if the execution of the specified action
   *   failed for any reason.
   * @throws IOException if there was an I/O error, e.g. retrieving metadata or
   *   setting outputs read-only.
   * @throws InterruptedException if the thread was interrupted.
   * @throws TargetOutOfDateException a subtype of ActionExecutionException if
   *   the target is not up to date and we are not allowed to execute the action
   *   making it up to date.
   */
  @ConditionallyThreadSafe
  protected void executeActionIfNeeded(Action action)
      throws InterruptedException, ActionExecutionException, IOException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }

    String error = badActions.get(action);
    if (error != null) {
      actionExecutor.reportError(error, action);
    }

    Token token;

    int progressIncrement = action.estimateWorkload();

    // The dependencyChecker is assumed to be thread-safe - thus there is no
    // longer any need to synchronize on it.
    if (LOG_FINEST) {
      LOG.finest(indent("Checking whether to make " + action));
    }

    profiler.startTask(ProfilerTask.ACTION_CHECK, action);
    long actionStartTime = Profiler.nanoTimeMaybe();
    try {
      Collection<Artifact> missingInputs = dependencyChecker.getMissingInputs(action);
      if (!missingInputs.isEmpty()) {
        ImmutableList.Builder<Label> rootCauses = ImmutableList.builder();

        for (Artifact missingInput : missingInputs) {
          reporter.handle(Event.error(action.getOwner().getLocation(),
              String.format("missing input file '%s'", missingInput.prettyPrint())));
          if (missingInput.getOwner() != null) {
            rootCauses.add(missingInput.getOwner());
          }
          recordMissingArtifact(missingInput);
        }

        // Do not report the error again as getMissingInputs() already reported each missing item.
        throw new ActionExecutionException(missingInputs.size() + " input file(s) do not exist",
            action, rootCauses.build(), false);
      }

      // Don't pass a reporter to DependencyChecker if no-one's listening.
      token = dependencyChecker.needToExecute(action, explain ? reporter : null);
    } finally {
      profiler.completeTask(ProfilerTask.ACTION_CHECK);
    }

    if (token == null) {
      boolean eventPosted = false;
      if (action.getActionType().isMiddleman()
          && action.getActionType() != MiddlemanType.TARGET_COMPLETION_MIDDLEMAN) {
        eventBus.post(new ActionStartedEvent(action, actionStartTime));
        eventBus.post(new ActionCompletionEvent(action, action.describeStrategy(executor)));
        eventPosted = true;
      }
      if (LOG_FINEST) {
        LOG.finest(indent("Don't need to make " + action));
      }

      if (action instanceof NotifyOnActionCacheHit) {
        NotifyOnActionCacheHit notify = (NotifyOnActionCacheHit) action;
        notify.actionCacheHit(executor);
      }

      if (!eventPosted) {
        eventBus
            .post(new CachedActionEvent(action, actionStartTime));
      }

      synchronized (workCompleted) {
        workCompleted.addAndGet(progressIncrement);
        workSavedByActionCache.addAndGet(progressIncrement);
      }
      return;
    } else if (dependencyChecker.isActionExecutionProhibited(action)) {
      // We can't execute an action (e.g. because --check_???_up_to_date option was used). Fail
      // the build instead.
      synchronized (reporter) {
        TargetOutOfDateException e = new TargetOutOfDateException(action);
        reporter.handle(Event.error(e.getMessage()));
        recordExecutionError();
        throw e;
      }
    }

    actionExecutor.createOutputDirectories(action);

    if (LOG_FINE) {
      LOG.fine(indent("Making " + action));
    }
    String message = action.getProgressMessage();
    if (message != null) {
      // Prepend a progress indicator.
      synchronized (workCompleted) {
        /*
         * Since workCompleted is always incremented (by a positive number), and since
         * we are computing the percent complete *and* reporting it to the reporter
         * in a synchronized block, and since the reporter method calls another reporter method
         * that is synchronized, progress monotonicity should be ensured.
         *
         * The synchronized block guarantees that no other thread reports progress between the
         * calls to longValue() and info().
         */
        if (!isBuilderAborting()) {
          reporter.startTask(null, getProgressPercentageDescription() + message);
        }
      }
    }
    try {
      actionExecutor.prepareScheduleExecuteAndCompleteAction(action, token, fileCache,
          dependencyChecker.getMetadataHandler(),
          executor == null
              ? null
              : ActionInputHelper.actionGraphMiddlemanExpander(executor.getActionGraph()),
          actionStartTime);
      workCompleted.addAndGet(progressIncrement);
    } finally {
      if (message != null) {
        /* See comment above about monotonicity. */
        synchronized (workCompleted) {
          reporter.finishTask(null, getProgressPercentageDescription() + message);
        }
      }
    }
  }

  protected abstract void scheduleAndExecuteAction(Action action,
      ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException;

  /** Indents a log message by the specified depth. */
  protected final String indent(String message) {
    return StringUtilities.indent(message, fileStack.size());
  }

  /**
   * Returns the amount of work saved as a result of never being scheduled by
   * the builder (i.e. work that never even gets considered for execution or
   * action cache checking).
   */
  protected long getWorkSkippedByBuilder() {
    return 0;
  }

  /**
   * Calculated the progress percentage.
   * The IDEs require that the percentage progress always increase
   * monotonically.  To ensure this, the caller of this method
   * MUST HOLD THE SYNCHRONIZATION LOCK on the workCompleted object
   * when this method is called, and MUST KEEP IT HELD until the
   * message has been printed.
   * See comments in executeActionIfNeeded() about monotonicity.
   */
  protected String getProgressPercentageDescription() {
    long workCompletedNow = workCompleted.longValue();
    long workCachedNow = workSavedByActionCache.longValue();
    long realWorkCompleted = workCompletedNow - workCachedNow;
    long realWorkLeft = workloadTotal - workCachedNow - getWorkSkippedByBuilder();

    // Previously we used to use this formula, which estimates
    // the percentage completion compared with a full (from-clean) build:
    //    (workloadTotal == 0) ? 100 : ((100 * workCompletedNow) / workloadTotal);
    // Now we use this formula instead, which estimates the percentage
    // completion of this build:
    int percentOfRealWork = (realWorkLeft == 0L)
        ? 100 :
        (int) ((100L * realWorkCompleted) / realWorkLeft);
    return "[" + percentOfRealWork + "%] ";
  }

  @Override
  public String toString() {
    return "AbstractBuilder";
  }

  /**
   * Register a missing artifact with the master log during the build.
   *
   * @param problem the missing artifact.
   */
  protected void recordMissingArtifact(Artifact problem) {
    Preconditions.checkNotNull(problem);
    Label label = problem.getOwner();
    eventBus.post(new MissingArtifactEvent(label));
  }

  /**
   * Record an action which failed to execute with the master log.
   *
   * @param action the action which failed
   */
  protected void recordActionFailure(Action action) {
    eventBus.post(new ActionFailedEvent(action));
  }

  protected MissingInputFileException getMissingInputFileException(Artifact artifact) {
    ActionOwner scapegoat = null;
    if (artifact.getOwner() == null) {
      return new MissingInputFileException(
          String.format("missing input file '%s'", artifact.getPath().getPathString()),
          null);
    }

    for (DependentAction dependentAction :
        dependencyChecker.getActionGraphForBuild().getDependentActions(artifact)) {
      if (dependentAction.getAction() == null ||
          dependentAction.getAction().getOwner() == null) {
        continue;
      }
      scapegoat = dependentAction.getAction().getOwner();
      break;
    }

    if (scapegoat == null) {
      return new MissingInputFileException(
          String.format("missing input file '%s'", artifact.getOwner()),
          null);
    } else {
      return new MissingInputFileException(
          String.format("%s: missing input file '%s'",
              scapegoat.getLabel(), artifact.getOwner()),
          scapegoat.getLocation());
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
}
