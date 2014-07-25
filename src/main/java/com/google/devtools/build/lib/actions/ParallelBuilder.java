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
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.Action.MiddlemanType;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadSafe;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Deque;
import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * A parallel Builder implementation.
 *
 * IMPORTANT NOTE: logging causes the creation of complex strings which are
 * typically not used.  This is an extreme performance problem, causing
 * measured slow-downs by a factor of 10 on no-op builds.  Add logging
 * judiciously; in performance-critical code, logging statements MUST be
 * conditional to a log-level test.
 */
@ConditionallyThreadCompatible
// ThreadCompatible provided that you don't try
// to run two builds in the same workspace concurrently.
public class ParallelBuilder extends AbstractBuilder implements ResourceManager.ResourceListener {

  /*
   * Signals and Interrupts
   * ======================
   *
   * The handling of signals is somewhat complex.  Here is the sequence of
   * events that occurs when the user types Control-C or otherwise sends SIGINT
   * to a Blaze process. This only describes interruptions in the execution
   * phase. Interruptions in the loading phase are described in
   * LabelVisitor.java .
   *
   * (1) User types Control-C.  Terminal delivers SIGINT to all processes in
   *      the foreground process-group.  In batch mode, this is the Blaze JVM
   *      and any subprocesses forked by currently-running actions.  In
   *      client/server mode, this just the Blaze client process, which passes
   *      it on to just the Blaze JVM.
   *
   * (2)  JVM's schedules special signal handling thread to run the
   *      application's SIGINT handler.
   *
   * It is not defined in which order the various processes respond to signal
   * delivery, and in client/server mode, the subprocesses won't receive any
   * signal from terminal Therefore there are two possible sequences of events
   * at this point, (a) and (b).
   *
   * (3a) The Java signal handler runs, and posts an interrupt to the
   *      application's main thread, in which the Builder runs.
   *
   * (4a) If the Builder is not blocked, i.e. it's visiting targets and doing
   *      dependency checking, then it may poll its thread's interrupted
   *      status, and iff interrupted, throws an InterruptedException up to
   *      the caller (but see note 6 below).  More often than not, however,
   *      the builder is blocked waiting for the ExecutionService.
   *
   * (5a) If the Builder is blocked in ExecutionService.take(), that wait is
   *      aborted and throws an InterruptedException, which is caught, causing
   *      the ExecutionService to be shut down, posting an interrupt
   *      to each worker-thread in its pool.
   *
   *      If the actions being executed by each worker-thread support
   *      interruptibility, they will exit early; otherwise they will run to
   *      completion.  (Notably, io.base.shell.Command does support
   *      early-termination when interrupted.  It doesn't throw
   *      InterruptedException, though; it merely preserves the worker-thread's
   *      interrupted status.)
   *
   *      Any job failures during shutdown will be quietly ignored.
   *      Once the ExecutionService shutown is complete, the
   *      InterruptedException is re-thrown up to the caller.
   *
   * If the Java signal handler does not run in a timely manner (as often
   * happens), one or more of the subprocesses forked by currently-executing
   * actions will fail due to the SIGINT.  In that case:
   *
   * (3b) One or more of the subprocesses fails due to SIGINT, causing the
   *      io.base.shell.Command (typically) to fail, and thence the execution
   *      of the action to fail.  The action then raises an
   *      ActionExecutionException. This is handled just like a normal action
   *      failure, causing the ParallelBuilder to either immediately rethrow
   *      the ActionExecutionException as a BuildFailedException OR if
   *      if --keep-going is set, suppress the interrupt and mark the build as
   *      failed and eventually raise a new BuildFailedException after all
   *      actions have been evaluated.
   *      Since the application still has not been notified of the SIGINT, it
   *      has no idea that a request to stop has been made.
   *
   * And then in both cases:
   *
   * (6)  The shutdownThreadPool method waits for all subprocesses to
   *      terminate, whether abruptly or normally.  This can take several
   *      seconds.  If the user is impatient and delivers additional signals to
   *      the JVM, these are delivered to the subprocesses, and additional
   *      interrupts are posted to the thread, again in no particular order.
   *
   *      It's important to note that in client/server mode, the Blaze client
   *      will send a SIGINT to the server the first two times it receives
   *      a SIGINT. The third time, however, it will send a SIGKILL.
   *      This way we allow users to work around "stuck" tests or genrules
   *
   *
   * For further reading about UNIX signals and process-groups, see "Advanced
   * Programming in the UNIX Environment", W. Richard Stephens.  For
   * information about Java thread interrupts, see the documentation at
   * java.lang.Thread, and also
   * http://www-128.ibm.com/developerworks/java/library/j-jtp05236.html.
   */

  //--- Local classes ----------------------------------------------------------

  /**
   * The ExecuteBuildAction class encapsulates an Action and its priority.
   * It is used to schedule actions and to execute actions via an Executor.
   */
  private class ExecuteBuildAction implements Callable<Action> {
    private final Action action;

    ExecuteBuildAction(Action action) {
      this.action = action;
    }

    /**
     * Note: runs in worker thread!
     */
    @Override
    @ThreadCompatible
    public Action call() throws ActionExecutionException {
      statusReporter.setPreparing(action);
      getProfiler().startTask(ProfilerTask.ACTION, action);
      try {
        try {
          executeActionIfNeeded(action);
        } catch (IOException e) {
          actionExecutor.reportError(e, action);
        } catch (InterruptedException e) {
          // Don't let InterruptedException propagate: exceptions thrown by
          // call() may be re-thrown by the Builder, but it seems wrong to throw
          // an InterruptedException belonging to a different thread.
          throw new ActionExecutionException("Execution interrupted", e, action, false);
        }
      } catch (ActionExecutionException e) {
        getProfiler().logException(e);
        throw e;
      } finally {
        statusReporter.remove(action);
        getProfiler().completeTask(ProfilerTask.ACTION);
      }
      return action;
    }
  }

  //--- Fields -----------------------------------------------------------------

  // See IMPORTANT NOTE about performance at class comment.
  private static final Logger LOG =
      Logger.getLogger(ParallelBuilder.class.getName());

  /** This service encapsulates the work queue and the thread pool. */
  private ThreadPoolExecutor actionExecutionService;

  /** This service encapsulates the completion queue. */
  private ExecutorCompletionService<Action> actionCompletionService;

  /** 'finished' gets set to true when the build request is completed. */
  private boolean finished;

  /**
   * 'buildFailed' starts out false and gets set to true if an action
   * failed and we're using keepGoing=true.  (In keepGoing=false mode,
   * we just propagate the exception out.)
   */
  private boolean buildFailed;

  private final int numWorkerThreads;

  private final int progressReportInterval;

  private final ResourceManager resourceManager = ResourceManager.instance();

  // Number of threads blocked by the resource acquisition.
  private final AtomicInteger blockedCount = new AtomicInteger();

  private long start; // Start time of the build, ns.
  private final boolean showBuilderStats;

  private final ActionExecutionStatusReporter statusReporter;

  /**
   * Some inputs might be allowed to be missing.
   */
  private final Predicate<PathFragment> allowedMissingInputs;

  protected DependentActionGraph dependencyGraph;

  // Current number of actions that have exactly one unbuilt dependency.
  private int singleDepCount = 0;

  //--- Public and protected methods -------------------------------------------

  /**
   * Constructor.  This constructs the thread pool and populates it with the
   * appropriate number of worker threads.
   *
   * @param reporter          Specifies how to report progress messages and
   *                          error messages.
   * @param dependencyChecker Specifies how to determine whether actions need
   *                          to be (re-)executed.
   * @param numWorkerThreads  Specifies the number of worker threads to use.
 *                          -1 means no limit; in this case a new worker
 *                          thread will be created whenever a new task
   * @param fileCache per-build action-input-metadata file cache.
   */
  public ParallelBuilder(Reporter reporter, EventBus eventBus,
      DependencyChecker dependencyChecker, int numWorkerThreads,
      int progressReportInterval, boolean showBuilderStats, boolean keepGoing,
      Path outputRoot, ActionInputFileCache fileCache,
      Predicate<PathFragment> allowedMissingInputs) {
    super(reporter, eventBus, dependencyChecker, keepGoing, outputRoot, fileCache);

    this.dependencyGraph = null;
    this.numWorkerThreads = numWorkerThreads;
    this.progressReportInterval = progressReportInterval;
    this.statusReporter = ActionExecutionStatusReporter.create(reporter);
    this.eventBus.register(statusReporter);
    this.showBuilderStats = showBuilderStats;
    this.allowedMissingInputs = allowedMissingInputs;
    constructThreadPool();
  }

  @Override
  protected void buildArtifactsHook(Set<Artifact> artifactSet, DependentActionGraph forwardGraph,
      ModifiedFileSet modified, Set<Artifact> builtArtifacts)
      throws BuildFailedException, InterruptedException, TestExecException {
    dependencyGraph = dependencyChecker.getActionGraphForBuild();
    // This is redundant in an incremental build, since actions were already
    // marked stale in IncrementalDependencyChecker.
    forwardGraph.markActionsStale(dependencyGraph.getActions());
    resourceManager.resetResourceUsage();
    resourceManager.addListener(this);

    if (!fileStack.isEmpty()) {
      throw new AssertionError(fileStack.toString());
    }

    LOG.info("Starting parallel build");
    finished = false;
    buildFailed = false;
    start = BlazeClock.nanoTime();

    try {
      // All actions are now stored in the graph; prepare them for execution.
      processGraphActions(artifactSet, builtArtifacts);

      // Wait for the worker threads to do their work.
      waitQueueLoop(keepGoing, builtArtifacts); // throws ActionExecutionException,
                                                //        InterruptedException
    } finally {
      resourceManager.removeListener(this);
      // Release any thread locks that might still exist if the build was aborted.
      resourceManager.resetResourceUsage();
    }

    // Release thread pool-related resources.  This is intentionally NOT called
    // in the case where waitQueueLoop throws an exception; it is required only
    // in the case of normal completion.  Yes, this is confusing.
    terminateThreadPool();

    LOG.info("Finished parallel build");

    if (buildFailed) {
      throw new BuildFailedException(); // catcher should print no error message
    }
  }

  private void processGraphActions(Set<Artifact> artifactSet,
                                   Set<Artifact> builtArtifacts)
      throws BuildFailedException, InterruptedException {

    Preconditions.checkNotNull(dependencyGraph);

    // The top-level action isn't stored with the other actions because its
    // parent action is null.
    enqueueIfReady(dependencyGraph.getTopLevelAction());

    // Source files aren't queued but don't depend on any other actions,
    // so they are built beforehand in a separate step.
    buildAllSourceFiles(artifactSet, builtArtifacts);

    // Now the actions in the graph are ready to run.
    Iterator<DependentAction> actionIterator = dependencyGraph.getDependentActions().iterator();

    // This traverses the actions in the graph, enqueueing actions which are
    // ready and marking source files built for their input actions.

    while (actionIterator.hasNext()) {
      DependentAction actionToBuild = actionIterator.next();
      enqueueIfReady(actionToBuild);

      for (Artifact input : actionToBuild.getAction().getInputs()) {
        if (input.isSourceArtifact()) {
          builtOneInputOf(actionToBuild, null);
        }
      }
    }
  }

  private void buildAllSourceFiles(Set<Artifact> artifactSet, Set<Artifact> builtArtifacts)
      throws BuildFailedException, InterruptedException {

    for (Artifact input : artifactSet) {
      if (input.isSourceArtifact()) {
        buildSourceFile(input, dependencyGraph.getTopLevelAction(), builtArtifacts);
      }
    }
    // At this point the dependencyGraph contains every non-source input artifact
    // and all associated generating actions.
    // Presence of each can be asserted, but doing so is slow.
  }

  private void buildSourceFile(Artifact file, DependentAction topLevelRequest,
                               Set<Artifact> builtArtifacts)
      throws BuildFailedException, InterruptedException {
    if (checkInputExistence(file)) {
      builtArtifacts.add(file);
    }
    // Immediately mark source artifacts as "built", reducing
    // "unbuiltArtifacts" count for the topLevelRequest.
    builtOneInputOf(topLevelRequest, null);
  }

  /**
   * Checks for existence of an input artifact. If the artifact does not exist, report that none
   * of the transitively dependent actions are executed.
   *
   * @param input the input artifact to check
   * @return true if the input exists; false if it does not but is allowed to be missing or
   *     {@link #keepGoing} is true; throws otherwise
   * @throws InterruptedException if the build was interrupted while shutting down the thread pool
   * @throws MissingInputFileException if a artifact is AWOL. This is a fatal error
   */
  private boolean checkInputExistence(Artifact input)
      throws InterruptedException, MissingInputFileException {
    if (dependencyChecker.artifactExists(input)) {
      return true;
    }

    if (allowedMissingInputs.apply(input.getExecPath())) {
      return false;
    }

    recordExecutionError();

    // Mark dependent actions as not-executed.
    if (input.getOwner() != null) {
      for (DependentAction dependentAction :
          dependencyChecker.getActionGraphForBuild().getDependentActions(input)) {
        if (!dependentAction.isVirtualCompletionAction()) {
          reportNotExecutedAction(dependentAction.getAction(), ImmutableList.of(input.getOwner()));
        }
      }
    }

    // Report missing file (with --keep_going) or throw.
    MissingInputFileException missingInputException = getMissingInputFileException(input);
    recordMissingArtifact(input);
    if (keepGoing) {
      // can't use printError here, since we have a null Action.
      buildFailed = true;
      reporter.error(missingInputException.getLocation(), missingInputException.getMessage());
      return false;
    } else {
      shutdownThreadPool();
      throw missingInputException;
    }
  }

  @Override
  public String toString() {
    return "ParallelBuilder";
  }

  //--- Private methods -------------------------------------------------------

  /** Construct the thread pool and work/completion queues. */
  private void constructThreadPool() {
    /*
     * Figure out how many threads to use, and whether to
     * reclaim threads which are not used for a long time,
     * and what kind of work queue to use.
     */
    int minThreads, maxThreads;
    long threadTimeoutInSeconds;
    BlockingQueue<Runnable> workQueue;
    if (numWorkerThreads > 0) {
      // Use exactly the specified number of threads
      minThreads = maxThreads = numWorkerThreads;
      threadTimeoutInSeconds = 0L;
      workQueue = new LinkedBlockingQueue<Runnable>();
    } else if (numWorkerThreads == -1) {
      // Use an unbounded number of threads,
      // but reclaim threads which have not been used for 60 seconds.
      minThreads = 1; // Don't let this be zero: ThreadPoolExecutor goes into a
                      // busy wait.
      maxThreads = Integer.MAX_VALUE;
      threadTimeoutInSeconds = 60L;
      // In this case, we never want to queue tasks;
      // we want to always create a new thread.
      // So use a zero-capacity queue.
      workQueue = new SynchronousQueue<Runnable>();
    } else {
      throw new IllegalArgumentException("invalid numWorkerThreads: " + numWorkerThreads);
    }

    actionExecutionService =
        new ThreadPoolExecutor(minThreads, maxThreads, threadTimeoutInSeconds,
            TimeUnit.SECONDS, workQueue,
            new ThreadFactoryBuilder().setNameFormat("ParallelBuilder %d").build());

    actionCompletionService = new ExecutorCompletionService<Action>(actionExecutionService);

    queued = dequeued = streamingActionsRunning = 0;
  }

  /**
   * Manually releases all execution service resources to make sure that
   * there are no references left to the parallel builder, action graph and
   * dependency checker. Method assumes that it is called after all execution
   * threads successfully completed their work so shutting down execution service
   * will just release resources.
   *
   * This method should not be used to terminate active worker threads -
   * shutdownThreadPool() should be used instead.
   *
   * NB! Without this code, due to circular references every build request will
   * create a new ParallelBuilder instance without releasing the resources used
   * by the old instance.
   */
  private void terminateThreadPool() {
    // TODO(bazel-team): (2010) First (short) investigation attempt to explain why explicit
    // shutdown command is needed at all did not reveal much. However this call
    // clearly resolves the issue so there must be something. It is also highly
    // advisable to review all inner classes defined for the ParallelBuilder to
    // see which one of them can use static modifier to further reduce dependencies.
    actionExecutionService.shutdownNow();

    // TODO(bazel-team): (2010) According to the profiler without manually setting the
    // execution service reference to null there is usually 1 extra copy of
    // the ParallelBuilder (almost) always hanging around even after forced
    // garbage collection. While the cause of this is not immediately clear,
    // setting actionExecutionService/actionCompletionService to null resolved
    // that issue. It still need to be revisited to find out all details.
    actionExecutionService = null;
    actionCompletionService = null;
  }

  /**
   * Shut down the thread pool and work/completion queues.  Each worker thread
   * is interrupted to try to induce an early shutdown.
   *
   * @param timeoutSec how long to wait for thread pool completion.
   *   BEWARE: Setting this less than Integer.MAX_VALUE is dangerous, as that
   *   it may allow for threads to remain active. Only do so if the blaze server
   *   is going to die anyway.
   * @throws InterruptedException iff the user interrupted the shutdown.
   */
  @ThreadSafe
  private void shutdownThreadPool(int timeoutSec) throws InterruptedException {
    LOG.info("Killing unfinished jobs...");
    // Shut down immediately, interrupting worker threads (which causes any
    // blocked on io.base.shell.Command to terminate their subprocesses).
    actionExecutionService.shutdownNow();
    try {
      awaitTermination(timeoutSec);
    } finally {
      actionExecutionService = null;
      actionCompletionService = null;
    }
  }

  private void shutdownThreadPool() throws InterruptedException {
    shutdownThreadPool(Integer.MAX_VALUE);
  }

  /**
   * Wait for all subprocesses to complete, unless two interrupts are received,
   * in which case the function returns without waiting for completion.  A
   * warning is issued after the first interrupt.
   *
   * @param timeoutSec how long to wait for thread pool completion.
   *   BEWARE: Setting this less than Integer.MAX_VALUE is dangerous, as that
   *   it may allow for threads to remain active. Only do so if the blaze server
   *   is going to die anyway.
   * @throws InterruptedException if the any user interrupts were received
   *   while waiting for the shutdown.  A single interrupt is not thrown
   *   immediately but deferred until the wait is completed; a second interrupt
   *   causes an immediate throw.
   */
  private void awaitTermination(int timeoutSec) throws InterruptedException {
    boolean wasInterrupted = Thread.currentThread().isInterrupted();
    for (;;) {
      try {
        if (!actionExecutionService.awaitTermination(timeoutSec, TimeUnit.SECONDS)) {
          reporter.warn(null, String.format("Worker tasks still active after %d seconds: " +
                                            "exiting anyway", timeoutSec));
        }
        break;
      } catch (InterruptedException e) {
        // See note (6) at "Signals and Interrupts".
        statusReporter.warnAboutCurrentlyExecutingActions();
        wasInterrupted = true;
      }
    }
    if (wasInterrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedException(); // i.e. one interrupt was received
    }
  }

  /**
   * The number of actions inserted onto the work queue.
   * This is similar to Java 1.5's threadPoolExecutor.getTaskCount(),
   * except that it is exact rather than approximate, and it does
   * not suffer from the Java 1.5 bug which caused the spec for
   * getTaskCount to be loosened in Java 1.6:
   * http://bugs.sun.com/bugdatabase/view_bug.do?bug_id=6448274
   */
  private long queued;

  /** The number of actions removed from the completed action queue. */
  private long dequeued;

  /** The number of possibly streaming actions currently executing. */
  private long streamingActionsRunning;

  /**
   * True iff we are currently executing an exclusive action.
   * An exclusive action is one which is not run concurrently
   * with another action. If an action and its reverse and forward
   * transitive closures comprise the entire set of actions, it will
   * be exclusive.
   *
   * Exclusive actions may always be streamed safely without
   * output from other actions being interleaved.
   */
  private boolean exclusiveAction = false;

  /**
   * Loop to get completed actions from the completed action queue,
   * and to put newly runnable actions on the work queue.
   * This runs in parallel with the worker threads which are taking runnable
   * actions from the work queue, executing them, and putting them on
   * the completed actions queue.
   *
   * Simultaneously, populate set of successfully built artifacts (if it is not
   * <i>null</i>).
   */
  private void waitQueueLoop(boolean keepGoing, Set<Artifact> builtArtifacts)
      throws BuildFailedException, InterruptedException, TestExecException {
    exclusiveAction = (queued - dequeued) == 1;
    while (!finished) {
      Action completedAction;
      ExecutionException exception;
      try {
        Pair<Action, ExecutionException> pair = getNextCompletedAction(keepGoing);
        completedAction = pair.first;
        exception = pair.second;
      } catch (BuildFailedException e) {
        if (!e.isCatastrophic() && !Iterables.isEmpty(e.getRootCauses())) {
          reportNotExecutedAction(e.getAction(), e.getRootCauses());
        }
        throw e;
      }
      Artifact problem = null;
      if (exception != null) {
        problem = completedAction.getPrimaryOutput();
        Iterable<Label> causes = ActionExecutionException.rootCausesFromAction(completedAction);
        if (exception.getCause() instanceof ActionExecutionException) {
          ActionExecutionException cause = (ActionExecutionException) exception.getCause();
          if (!Iterables.isEmpty(cause.getRootCauses())) {
            causes = cause.getRootCauses();
          }
        }

        reportNotExecutedAction(completedAction, causes);
      }
      boolean noneQueued = (queued - dequeued) == 0;
      markActionAsDone(completedAction, problem);
      exclusiveAction = noneQueued && ((queued - dequeued) == 1);

      // Artifacts are successfully built only if action does not have
      // an associated exception.
      if (builtArtifacts != null && exception == null) {
        builtArtifacts.addAll(completedAction.getOutputs());
        // Note that this removal is not thread-safe, but is only being done in the main thread.
        forwardGraph.markActionNotStale(completedAction);
      }
    }
  }

  // Since ArrayDeque does not allow null values, we use a marker object to see when we have
  // finished traversing an action's children.
  private static final ActionMetadata DUMMY_MARKER = new ActionMetadata() {

    @Override
    public String getProgressMessage() {
      throw new UnsupportedOperationException();
    }

    @Override
    public ActionOwner getOwner() {
      throw new UnsupportedOperationException();
    }

    @Override
    public String getMnemonic() {
      throw new UnsupportedOperationException();
    }

    @Override
    public String prettyPrint() {
      throw new UnsupportedOperationException();
    }

    @Override
    public String describeStrategy(Executor executor) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean inputsKnown() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean discoversInputs() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Iterable<Artifact> getInputs() {
      throw new UnsupportedOperationException();
    }

    @Override
    public int getInputCount() {
      throw new UnsupportedOperationException();
    }

    @Override
    public ImmutableSet<Artifact> getOutputs() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Artifact getPrimaryInput() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Artifact getPrimaryOutput() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Iterable<Artifact> getMandatoryInputs() {
      throw new UnsupportedOperationException();
    }

    @Override
    public String getKey() {
      throw new UnsupportedOperationException();
    }

    @Override
    public String describeKey() {
      throw new UnsupportedOperationException();
    }};

  /**
   * In the case of a Blaze bug that causes a deadlock during execution (no actions are ready to be
   * executed, and no actions are running), this method iterates over the entire forward graph and
   * finds all cycles, under the assumption that the only way to cause such a deadlock is via a
   * cycle. This is an expensive traversal (O(edges)), but only occurs when Blaze is about to crash
   * in this rare case.
   */
  private String findCycles() {
    // Keep track of actions we've visited to save work.
    Set<ActionMetadata> processed = Sets.newHashSet();
    // String with information on all cycles.
    StringBuilder builder = new StringBuilder();

    // Iterate over all actions in forward graph.
    for (DependentAction currentTop : dependencyGraph.getDependentActions()) {
      // Keep track of all actions visited on this path through the graph.
      Deque<ActionMetadata> stack = new ArrayDeque<ActionMetadata>();
      // Actions that remain to be visited on this path. Children are grouped together with
      // DUMMY_MARKER as a separator, so that we can tell when all of a node's children have been
      // visited and it can be popped off the stack.
      Deque<ActionMetadata> toVisit = new ArrayDeque<ActionMetadata>();
      toVisit.add(Preconditions.checkNotNull(currentTop.getAction(), currentTop));

      while (!toVisit.isEmpty()) {
        ActionMetadata action = toVisit.pop();

        if (action == DUMMY_MARKER) {
          // All children of the top element of the stack have been visited, so we are done with it.
          stack.pop();
          continue;
        }

        if (stack.contains(action)) {
          builder.append("\nFound cycle in action graph:\n");
          // The "*" indicates the repeated action in the cycle.
          builder.append(" * " + action.prettyPrint() + "\n");
          for (ActionMetadata stackAction : stack) {
            builder.append((action.equals(stackAction) ? " * " : "   ")
                + stackAction.prettyPrint() + "\n");
          }
          continue;
        }

        if (!processed.add(action)) {
          continue;
        }

        // Put separator into toVisit, so we know when this action's children are finished.
        toVisit.push(DUMMY_MARKER);
        stack.push(action);
        // Enqueue all actions that this action depends on.
        for (Artifact input : action.getInputs()) {
          ActionMetadata childAction = dependencyGraph.getActionGraph().getGeneratingAction(input);
          if (childAction == null) {
            continue;
          }
          toVisit.push(childAction);
        }
      }
    }
    return builder.toString();
  }

  /**
   * Get the next action from the completed actions queue.
   * This blocks until a worker thread has completed an action.
   *
   * @return A pair (action, null)      if the action was successful.
   *         A pair (action, exception) if the action failed
   *                                    and keepGoing is true.
   * @throws BuildFailedException       if an individual the action has failed.
   *                                    Suppressed if keepGoing.
   * @throws InterruptedException       if interrupted.
   */
  private Pair<Action, ExecutionException> getNextCompletedAction(boolean keepGoing)
      throws BuildFailedException, InterruptedException, TestExecException {
    try {
      if (dequeued >= queued) {
        throw new IllegalStateException(
            "Trying to dequeue more actions than were queued: would deadlock!"
            + findCycles());
      }

      Future<Action> nextCompletedAction;

      // Wait time (in seconds) to print out "Still waiting..." message.
      int waitTime = ActionExecutionStatusReporter.getWaitTime(progressReportInterval, 0);
      while ((nextCompletedAction = actionCompletionService.poll(waitTime, TimeUnit.SECONDS))
          == null) {
        // Avoid interleaving the "Still waiting" message with the streaming
        // output of a single test action.
        if (!(exclusiveAction && streamingActionsRunning == 1)) {
          // No jobs completed within waitTime.  Print the jobs still in flight.
          synchronized (workCompleted) { // See comments in AbstractBuilder about monotonicity.
            statusReporter.showCurrentlyExecutingActions(getProgressPercentageDescription());
          }
        }
        waitTime = ActionExecutionStatusReporter.getWaitTime(progressReportInterval, waitTime);
      }
      dequeued++; // take() succeeded; get() may still throw.
      Action completedAction = nextCompletedAction.get();
      doneTestMaybe(completedAction);
      getProfiler().logEvent(ProfilerTask.ACTION_BUILDER, completedAction);
      if (this.showBuilderStats) {
        reportStatistics(completedAction.prettyPrint());
      }
      LOG.fine("Got completed action from completion queue");
      return Pair.of(completedAction, null);
    } catch (InterruptedException e) {
      // See note (5a) at "Signals and Interrupts".
      //
      // We call shutdownThreadPool() but with an interrupt pending,
      // so shutdown will (almost) always throw InterruptedException.
      // Occasionally (race condition--presumably the job just finished at this
      // exact moment) it returns normally though, so in that case we throw the
      // InterruptedException ourselves.
      Thread.currentThread().interrupt(); // don't lose the interrupt!
      shutdownThreadPool(); // throws InterruptedException >99% of the time
      throw e; // very rarely reach here.
    } catch (ExecutionException e) {
      /* This indicates that the action failed. */
      return handleExecutionException(e, keepGoing);
    }
  }

  private Pair<Action, ExecutionException>
      handleExecutionException(ExecutionException exception, boolean keepGoing)
      throws BuildFailedException, InterruptedException, TestExecException {
    Throwable cause = exception.getCause();
    Throwable innerCause = cause.getCause();
    if (cause instanceof ActionExecutionException &&
        !(innerCause instanceof TestExecException)) {
      ActionExecutionException actionCause = (ActionExecutionException) cause;
      Action action = actionCause.getAction();
      recordActionFailure(action);
      if (keepGoing && !actionCause.isCatastrophe()) {
        doneTestMaybe(action);
        return Pair.of(action, exception);
      }
    }

    LOG.fine("Got exception from completion queue: " + cause.getMessage());
    /* Kill off any unfinished jobs and then throw an exception. */
    int timeoutSec = Integer.MAX_VALUE;
    if ((cause instanceof RuntimeException) || (cause instanceof Error)) {
      // It is safe to not hang forever here, since we will exit the blaze server regardless.
      // This is useful if the unexpected exception here causes other threads to block.
      timeoutSec = 100;
    }
    shutdownThreadPool(timeoutSec);
    BuilderUtils.rethrowCause(exception);
    throw new IllegalStateException(); // can't get here
  }

  private void doneTestMaybe(Action action) {
    if (action instanceof MayStream) {
      streamingActionsRunning--;
    }
  }

  /**
   * Mark the Action as having been done, and add any
   * dependent actions with no remaining unbuilt
   * dependencies to the work queue.
   *
   * PERFORMANCE CRITICAL!
   */
  private void markActionAsDone(Action completedAction, Artifact problem) {
    if (LOG_FINE) {
      LOG.fine(indent("Marking " + completedAction.prettyPrint() + " as done"));
    }

    if (problem != null) {
      recordActionFailure(completedAction);
    }

    for (Artifact output : completedAction.getOutputs()) {
      markArtifactAsDone(output, problem);
    }
  }

  private void reportNotExecutedAction(Action completedAction, Iterable<Label> rootCauses) {
    walkDependentActionsAndSignalNotExecuted(
        rootCauses, completedAction, Sets.<Action>newHashSet());
  }

  private void walkDependentActionsAndSignalNotExecuted(
      Iterable<Label> rootCauses, Action currentAction, Set<Action> actionsAlreadySeen) {
    Preconditions.checkNotNull(dependencyGraph);

    for (Artifact artifact : currentAction.getOutputs()) {
      Collection<DependentAction> deps =
          dependencyGraph.getArtifactDependenciesInternal(artifact);

      if (deps == null) {
        continue;
      }
      for (DependentAction d : deps) {
        Action notExecuted = d.getAction();
        if (notExecuted == null) {
          continue;
        }
        if (actionsAlreadySeen.add(notExecuted)
            && notExecuted.getActionType() != MiddlemanType.SCHEDULING_MIDDLEMAN) {
          actionExecutor.postActionNotExecutedEvents(notExecuted, rootCauses);
          walkDependentActionsAndSignalNotExecuted(
              rootCauses, notExecuted, actionsAlreadySeen);
        }
      }
    }
  }

  /**
   * Record that the given file has been built.
   *
   * <p>All {@link DependentAction}s will be notified, so they can decrease their unbuilt input
   * counters, and we can add those with no remaining unbuilt dependencies to the work queue.
   * Also, removes the Artifact from the dependenciesMap since we only need to notify its
   * dependents once.
   *
   * <p>PERFORMANCE CRITICAL!
   */
  private void markArtifactAsDone(Artifact file, Artifact problem) {
    if (LOG_FINER) {
      LOG.finer(indent("Marking " + file.prettyPrint() + " as done"));
    }

    Collection<DependentAction> dependencies = dependencyGraph.getAndRemoveArtifactMaybe(file);
    if (dependencies != null) {
      for (DependentAction dependent : dependencies) {
        builtOneInputOf(dependent, problem);
      }
    }
  }

  /**
   * Execute the specified action, acquiring any scheduling locks needed.
   * The caller is responsible for having already checked that we need to
   * execute it.  This function is responsible for profiling.
   *
   * This is thread-safe so long as you don't try to execute the same action
   * twice at the same time (or overlapping times).
   * May execute in a worker thread.
   *
   * @throws ActionExecutionException if the execution of the specified action
   *   failed for any reason.
   * @throws InterruptedException if the thread was interrupted.
   */
  @ConditionallyThreadSafe
  @Override
  protected void scheduleAndExecuteAction(Action action,
      ActionExecutionContext actionExecutionContext)
  throws ActionExecutionException, InterruptedException {
    ResourceSet resources = action.estimateResourceConsumption(executor);

    if (resources == null || resources == ResourceSet.ZERO) {
      statusReporter.setRunningFromBuildData(action, executor);
    } else {
      // If estimated resource consumption is null, action will manually call
      // resource manager when it knows what resources are needed.
      resourceManager.acquireResources(action, resources);
    }

    try {
      actionExecutor.executeActionTask(action, actionExecutionContext);
    } finally {
      if (resources != null && resources != ResourceSet.ZERO) {
        resourceManager.releaseResources(action, resources);
      }
    }
  }

  @Override
  protected long getWorkSkippedByBuilder() {
    return dependencyChecker.getWorkSavedByDependencyChecker();
  }

  /**
   * Tell the {@link DependentAction} that one of its inputs was built.
   *
   * <p>If all inputs have been built, add the action to the work queue.
   *
   * <p>PERFORMANCE CRITICAL!
   */
  private void builtOneInputOf(DependentAction dependentAction, Artifact problem) {
    Preconditions.checkState(dependentAction.getUnbuiltInputs() > 0);
    if (DependentActionGraph.LOG_FINEST) {
      LOG.finest(indent(String.format("Built one input of %s (%d to go)",
          dependentAction.prettyPrint(), dependentAction.getUnbuiltInputs())));
    }

    if (problem != null) {
      dependentAction.recordProblem(problem);
      // As elsewhere, do not propagate errors in scheduling middlemen upward.
      if (!isSchedulingMiddlemanAction(dependentAction.getAction())) {
        recordActionFailure(dependentAction.getAction());
      }
    }

    dependentAction.builtOneInput();
    if (dependentAction.getUnbuiltInputs() == 0) {
      singleDepCount--;
    }
    enqueueIfReady(dependentAction);
  }

  /**
   * If all of the inputs of an action have been built, add the action to the work queue.
   *
   * <p>PERFORMANCE CRITICAL!
   */
  private void enqueueIfReady(DependentAction dependentAction) {
    int pendingInputs = dependentAction.getUnbuiltInputs();
    Preconditions.checkState(pendingInputs >= 0);

    if (pendingInputs > 0) {
      if (pendingInputs == 1) {
        singleDepCount++;
      }
      return;
    }

    // We've built all the inputs (or at least tried to -- some may have
    // failed), so now this action is ready to run.
    Artifact problem = dependentAction.getLastProblemInput();
    if (dependentAction.isVirtualCompletionAction()) {
      if (problem != null) {
        buildFailed = true;
      } else {
        LOG.fine("All targets completed.");
      }
      finished = true;
    } else {
      Action action = Preconditions.checkNotNull(dependentAction.getAction());
      if (problem != null) {
        // Suppress error propagation for scheduling middlemen, since they
        // are used only to enforce scheduling and not build dependencies.
        markActionAsDone(action, isSchedulingMiddlemanAction(action) ? null : problem);
      } else {
        if (DependentActionGraph.LOG_FINE) {
          LOG.fine(indent("Putting action on work queue: " + action));
        }
        getProfiler().logEvent(ProfilerTask.ACTION_SUBMIT, action);
        actionCompletionService.submit(new ExecuteBuildAction(action));
        queued++;
        if (action instanceof MayStream) {
          streamingActionsRunning++;
        }
      }
    }
    // We reset this dependent Action here because we are done with it for this build. Note that
    // if we have --keep_forward_graph and this dependent action is toUpdate (in
    // StableDependentActionGraph) then this dependent action will be reset again in
    // StableDependentActionGraph#sync. As well, if the build fails before the associated action
    // is marked not stale in #waitQueueLoop, then there too the dependent action will be reset
    // again in sync(). Since #reset() is idempotent, this isn't a problem.
    dependentAction.reset();
  }

  private boolean isSchedulingMiddlemanAction(@Nullable Action action) {
    return action != null && action.getActionType().isSchedulingMiddleman();
  }

  @Override
  public void waiting(ActionMetadata owner) {
    // This approach assumes that actions acquire resources from a single thread.
    statusReporter.setScheduling(owner);
    blockedCount.incrementAndGet();
  }

  @Override
  public void acquired(ActionMetadata owner) {
    statusReporter.setRunningFromBuildData(owner, executor);
    blockedCount.decrementAndGet();
  }

  /**
   * Report builder statistic on every action completion event in following format:
   *
   * [time_since_start_of_the_buildArtifacts] s,
   * queue: [cumulative scheduled actions] / [cumulative completed actions],
   * threads: [queued tasks] / [active threads in the threadpool] /
   *          [threads, not blocked by scheduling lock] /
   *          [threads in ExecuteBuildAction.call()],
   * actions: [number of actions waiting on the last dependency before being scheduled]
   *          ([total number of actions]),
   * locked: [threads blocked by the scheduling lock due to resource contention],
   * >>> [action name]
   *
   * Action name will be printed only if at the time when action completed there
   * were no tasks in the queue - thus action was potentially on the critical path.
   */
  private void reportStatistics(String message) {
    double time = ((BlazeClock.nanoTime() - start) / 100000000) / 10.0;

    // Number of queued actions (either already assigned to the worker thread or
    // waiting for one.
    long queuedCount = queued - dequeued;

    // Active threads as reported by the thread pool.
    int activeCount = actionExecutionService.getActiveCount();

    // Number of threads, NOT blocked by the resource acquisition.
    int workingCount = activeCount - blockedCount.get();
    reporter.progress(null, String.format(
        "%s s, queue: %d / %d, threads: %d / %d / %d / %d, actions: %d (%d), locked: %d%s",
        /* time, queue */ new DecimalFormat("#.##").format(time), queued, dequeued,
        /* threads */ queuedCount, activeCount, workingCount, statusReporter.getCount(),
        /* actions */ singleDepCount, dependencyGraph.getRemainingActionCount(),
        /* locked */ blockedCount.get(), ((queuedCount <= activeCount) ? " >>> " + message : "")));
  }
}
