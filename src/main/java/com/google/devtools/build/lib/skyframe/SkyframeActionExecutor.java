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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.eventbus.EventBus;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.AbstractActionExecutor;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Action.MiddlemanType;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionCacheChecker.DepcheckerListener;
import com.google.devtools.build.lib.actions.ActionCacheChecker.Token;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.actions.ActionFailedEvent;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.MiddlemanExpander;
import com.google.devtools.build.lib.actions.ArtifactPrefixConflictException;
import com.google.devtools.build.lib.actions.CachedActionEvent;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.MapBasedActionGraph;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.TargetOutOfDateException;
import com.google.devtools.build.lib.actions.cache.Digest;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.concurrent.ExecutorShutdownUtil;
import com.google.devtools.build.lib.concurrent.Sharder;
import com.google.devtools.build.lib.concurrent.ThrowableRecordingRunnableWrapper;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
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

/** Skyframe-based action executor implementation. */
public final class SkyframeActionExecutor extends AbstractActionExecutor {
  private final AtomicReference<EventBus> eventBus;
  private final ResourceManager resourceManager;
  private ActionCacheChecker actionCacheChecker;
  private ConcurrentMap<Artifact, Metadata> undeclaredInputsMetadata = new ConcurrentHashMap<>();
  private final Profiler profiler = Profiler.instance();
  private DepcheckerListener depcheckerListener;

  // We keep track of actions already executed this build in order to avoid executing a shared
  // action twice. Note that we may still unnecessarily re-execute the action on a subsequent
  // build: say actions A and B are shared. If A is requested on the first build and then B is
  // requested on the second build, we will execute B even though its output files are up to date.
  // However, we will not re-execute A on a subsequent build.
  // We do not allow the shared action to re-execute in the same build, even after the first
  // action has finished execution, because a downstream action might be reading the output file
  // at the same time as the shared action was writing to it.
  // This map is also used for Actions that try to execute twice because they have discovered
  // headers -- the NodeBuilder tries to declare a dep on the missing headers and has to restart.
  // We don't want to execute the action again on the second entry to the NodeBuilder.
  private ConcurrentMap<Artifact, Pair<Action, FutureTask<Void>>> buildActionMap;

  // Errors found when examining all actions in the graph are stored here, so that they can be
  // thrown when execution of the action is requested. This field is set during each call to
  // findAndStoreArtifactConflicts, and is preserved across builds otherwise.
  private ImmutableMap<Action, Exception> badActionMap = ImmutableMap.of();
  private boolean keepGoing;
  private boolean hadExecutionError;
  private ActionInputFileCache perBuildFileCache;
  private ProgressSupplier progressSupplier;
  private ActionCompletedReceiver completionReceiver;
  private final AtomicReference<ActionExecutionStatusReporter> statusReporterRef;

  SkyframeActionExecutor(Reporter reporter, ResourceManager resourceManager,
      AtomicReference<EventBus> eventBus,
      AtomicReference<ActionExecutionStatusReporter> statusReporterRef) {
    super(reporter);
    this.resourceManager = resourceManager;
    this.eventBus = eventBus;
    this.statusReporterRef = statusReporterRef;
  }

  /**
   * Basic implementation of {@link MetadataHandler} that delegates to Skyframe for metadata and
   * caches missing source artifacts (which must be undeclared inputs: discovered headers) to avoid
   * excessive filesystem access. The discovered-header cache is available across actions.
   */
  // TODO(bazel-team): remove when include scanning is skyframe-native.
  private static class UndeclaredInputHandler implements MetadataHandler {
    private final ConcurrentMap<Artifact, Metadata> undeclaredInputsMetadata;
    private final MetadataHandler perActionHandler;

    UndeclaredInputHandler(MetadataHandler perActionHandler,
        ConcurrentMap<Artifact, Metadata> undeclaredInputsMetadata) {
      // Shared across all UndeclaredInputHandlers in this build.
      this.undeclaredInputsMetadata = undeclaredInputsMetadata;
      this.perActionHandler = perActionHandler;
    }

    @Override
    public Metadata getMetadataMaybe(Artifact artifact) {
      try {
        return getMetadata(artifact);
      } catch (IOException e) {
        return null;
      }
    }

    @Override
    public Metadata getMetadata(Artifact artifact) throws IOException {
      Metadata metadata = perActionHandler.getMetadata(artifact);
      if (metadata != null) {
        return metadata;
      }
      // Skyframe stats all generated artifacts on the assumption they may be outputs.
      Preconditions.checkState(artifact.isSourceArtifact(), artifact);
      metadata = undeclaredInputsMetadata.get(artifact);
      if (metadata != null) {
        return metadata;
      }
      FileStatus stat = artifact.getPath().stat();
      if (DigestUtils.useFileDigest(artifact, stat.isFile(), stat.getSize())) {
        metadata = new Metadata(stat.getLastModifiedTime(),
            DigestUtils.getDigestOrFail(artifact.getPath(), stat.getSize()));
      } else {
        metadata = new Metadata(stat.getLastModifiedTime(), null);
      }
      // Cache for other actions that may also include without declaring.
      Metadata oldMetadata = undeclaredInputsMetadata.put(artifact, metadata);
      FileAndMetadataCache.checkInconsistentData(artifact, oldMetadata, metadata);
      return metadata;
    }

    @Override
    public void setDigestForVirtualArtifact(Artifact artifact, Digest digest) {
      perActionHandler.setDigestForVirtualArtifact(artifact, digest);
    }

    @Override
    public void injectDigest(ActionInput output, FileStatus stat, byte[] digest) {
      perActionHandler.injectDigest(output, stat, digest);
    }

    @Override
    public boolean artifactExists(Artifact artifact) {
      return perActionHandler.artifactExists(artifact);
    }

    @Override
    public boolean isInjected(Artifact artifact) throws IOException {
      return perActionHandler.isInjected(artifact);
    }

    @Override
    public void discardMetadata(Collection<Artifact> artifactList) {
      // This input handler only caches undeclared inputs, which never need to be discarded
      // intra-build.
      perActionHandler.discardMetadata(artifactList);
    }
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
  void findAndStoreArtifactConflicts(Iterable<ActionLookupNode> actionLookupNodes)
      throws InterruptedException {
    ConcurrentMap<Action, Exception> temporaryBadActionMap = new ConcurrentHashMap<>();
    Pair<ActionGraph, SortedMap<PathFragment, Artifact>> result;
    result = constructActionGraphAndPathMap(actionLookupNodes, temporaryBadActionMap);
    ActionGraph actionGraph = result.first;
    SortedMap<PathFragment, Artifact> artifactPathMap = result.second;

    // Report an error for every derived artifact which is a prefix of another.
    // If x << y << z (where x << y means "y starts with x"), then we only report (x,y), (x,z), but
    // not (y,z).
    Iterator<PathFragment> iter = artifactPathMap.keySet().iterator();
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
            Exception exception = new ArtifactPrefixConflictException(pathI, pathJ,
                actionI.getOwner().getLabel(), actionJ.getOwner().getLabel());
            temporaryBadActionMap.put(actionI, exception);
            temporaryBadActionMap.put(actionJ, exception);
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
          Iterable<ActionLookupNode> nodes,
          ConcurrentMap<Action, Exception> badActionMap) throws InterruptedException {
    MutableActionGraph actionGraph = new MapBasedActionGraph();
    ConcurrentNavigableMap<PathFragment, Artifact> artifactPathMap = new ConcurrentSkipListMap<>();
    // Action graph construction is CPU-bound.
    int numJobs = Runtime.getRuntime().availableProcessors();
    // No great reason for expecting 5000 action lookup nodes, but not worth counting size of nodes.
    Sharder<ActionLookupNode> actionShards = new Sharder<>(numJobs, 5000);
    for (ActionLookupNode node : nodes) {
      actionShards.add(node);
    }

    ThrowableRecordingRunnableWrapper wrapper = new ThrowableRecordingRunnableWrapper(
        "SkyframeActionExecutor#constructActionGraphAndPathMap");

    ExecutorService executor = Executors.newFixedThreadPool(
        numJobs,
        new ThreadFactoryBuilder().setNameFormat("ActionLookupNode Processor %d").build());
    for (List<ActionLookupNode> shard : actionShards) {
      executor.execute(
          wrapper.wrap(actionRegistration(shard, actionGraph, artifactPathMap, badActionMap)));
    }
    boolean interrupted = ExecutorShutdownUtil.interruptibleShutdown(executor);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted) {
      throw new InterruptedException();
    }
    return Pair.<ActionGraph, SortedMap<PathFragment, Artifact>>of(actionGraph, artifactPathMap);
  }

  private static Runnable actionRegistration(
      final List<ActionLookupNode> nodes,
      final MutableActionGraph actionGraph,
      final ConcurrentMap<PathFragment, Artifact> artifactPathMap,
      final ConcurrentMap<Action, Exception> badActionMap) {
    return new Runnable() {
      @Override
      public void run() {
        Action previousAction = null;
        for (ActionLookupNode node : nodes) {
          for (Map.Entry<Artifact, Action> entry : node.getMapForConsistencyCheck().entrySet()) {
            Action action = entry.getValue();
            if (!action.equals(previousAction)) {
              // We have an entry for each <action, artifact> pair. Only try to register each action
              // once.
              previousAction = action;
              try {
                actionGraph.registerAction(action);
              } catch (ActionConflictException e) {
                Exception oldException = badActionMap.put(action, e);
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

  void prepareForExecution(Executor executor, boolean keepGoing,
      ActionCacheChecker actionCacheChecker) {
    setExecutorEngine(Preconditions.checkNotNull(executor));

    // Start with a new map each build so there's no issue with internal resizing.
    this.buildActionMap = Maps.newConcurrentMap();
    this.keepGoing = keepGoing;
    this.hadExecutionError = false;
    this.actionCacheChecker = Preconditions.checkNotNull(actionCacheChecker);
    // Don't cache possibly stale data from the last build.
    undeclaredInputsMetadata = new ConcurrentHashMap<>();
    this.depcheckerListener = DepcheckerListener.createListenerMaybe(reporter);
  }

  File getExecRoot() {
    return executorEngine.getExecRoot().getPathFile();
  }

  /**
   * Executes the provided action on the current thread.
   *
   * <p>For use from {@link ArtifactNodeBuilder} only.
   */
  protected void executeAction(Action action, FileAndMetadataCache graphFileCache)
      throws ActionExecutionException, InterruptedException {
    Exception exception = badActionMap.get(action);
    if (exception != null) {
      // If action had a conflict with some other action in the graph, report it now.
      reportError(exception, action);
    }
    Artifact primaryOutput = action.getPrimaryOutput();
    FutureTask<Void> actionTask = new FutureTask<Void>(new ActionRunner(action, graphFileCache));
    // Check to see if another action is already executing/has executed this node.
    Pair<Action, FutureTask<Void>> oldAction =
        buildActionMap.putIfAbsent(primaryOutput, Pair.of(action, actionTask));
    ActionExecutionStatusReporter statusReporter =
        Preconditions.checkNotNull(statusReporterRef.get());

    boolean waitingForShared = false;

    if (oldAction == null) {
      ResourceSet estimate = action.estimateResourceConsumption(executorEngine);
      statusReporter.setRunningFromBuildData(action);
      try {
        if (estimate != null) {
          // If estimated resource consumption is null, action will manually call
          // resource manager when it knows what resources are needed.
          resourceManager.acquireResources(action, estimate);
        }
        actionTask.run();
      } finally {
        // We don't expect to see any exceptions thrown from the try block, but it doesn't hurt
        // to put releaseResources() in a finally block.
        if (estimate != null) {
          resourceManager.releaseResources(action, estimate);
        }
        statusReporter.remove(action);
      }
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
      waitingForShared = true;
    }
    try {
      actionTask.get();
      if (waitingForShared || action.discoversInputs()) {
        // We wastefully check the output files again, so that this node can retrieve the data. If
        // it was waiting for the shared action to be executed, it did not actually find the output
        // data itself. If the action discovers inputs and this is the second run (after the action
        // declared its dependencies on undeclared inputs and the node builder restarted), we didn't
        // actually execute the action. Note that if the action actually was executed, there will be
        // no additional filesystem access here. This is a trade-off of filesystem access versus
        // memory -- we could keep the output data of every action through the entire execution
        // phase, but that might be too expensive. Until shown to be a bottleneck, these filesystem
        // accesses shouldn't be too bad.
        checkOutputs(action, graphFileCache);
      }
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

  /**
   * This method should be called if the builder encounters an error during
   * execution. This allows the builder to record that it encountered at
   * least one error, and may make it swallow its output to prevent
   * spamming the user any further.
   */
  protected void recordExecutionError() {
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
  protected boolean isBuilderAborting() {
    return hadExecutionError && !keepGoing;
  }

  public void setFileCache(ActionInputFileCache fileCache) {
    this.perBuildFileCache = fileCache;
  }

  private class ActionRunner implements Callable<Void> {
    private final Action action;
    private final FileAndMetadataCache graphFileCache;

    ActionRunner(Action action, FileAndMetadataCache graphFileCache) {
      this.action = action;
      this.graphFileCache = graphFileCache;
    }

    @Override
    public Void call() throws ActionExecutionException, InterruptedException {
      profiler.startTask(ProfilerTask.ACTION_CHECK, action);
      long actionStartTime = Profiler.nanoTimeMaybe();

      MetadataHandler metadataHandler =
          new UndeclaredInputHandler(graphFileCache, undeclaredInputsMetadata);
      Token token = actionCacheChecker.needToExecute(action, depcheckerListener, metadataHandler);
      profiler.completeTask(ProfilerTask.ACTION_CHECK);

      if (token == null) {
        boolean eventPosted = false;
        // Notify BlazeRuntimeStatistics about the action middleman 'execution'.
        if (action.getActionType().isMiddleman()
            && action.getActionType() != MiddlemanType.TARGET_COMPLETION_MIDDLEMAN) {
          postEvent(new ActionStartedEvent(action, actionStartTime));
          postEvent(new ActionCompletionEvent(
              action, actionGraph, action.describeStrategy(executorEngine)));
          eventPosted = true;
        }

        if (action instanceof NotifyOnActionCacheHit) {
          NotifyOnActionCacheHit notify = (NotifyOnActionCacheHit) action;
          notify.actionCacheHit(executorEngine);
        }

        // We still need to check the outputs so that output file data is available to the node.
        checkOutputs(action, graphFileCache);
        if (!eventPosted) {
          postEvent(new CachedActionEvent(action, actionGraph, actionStartTime));
        }

        return null;
      } else if (actionCacheChecker.isActionExecutionProhibited(action)) {
        // We can't execute an action (e.g. because --check_???_up_to_date option was used). Fail
        // the build instead.
        synchronized (reporter) {
          TargetOutOfDateException e = new TargetOutOfDateException(action);
          reporter.error(null, e.getMessage());
          recordExecutionError();
          throw e;
        }
      }

      String message = action.getProgressMessage();
      if (message != null) {
        reporter.startTask(null, prependExecPhaseStats(message));
      }

      createOutputDirectories(action);

      prepareScheduleExecuteAndCompleteAction(action, token,
            new DelegatingPairFileCache(graphFileCache, perBuildFileCache), metadataHandler,
            new MiddlemanExpander() {
              @Override
              public void expand(Artifact middlemanArtifact, Collection<? super Artifact> output) {
                // Legacy code is more permissive regarding "mm" in that it expands any middleman,
                // not just inputs of this action. Skyframe doesn't have access to a global action
                // graph, therefore this implementation can't expand any middleman, only the inputs
                // of this action.
                // This is fine though: actions should only hold references to their input
                // artifacts, otherwise hermeticity would be violated.
                output.addAll(graphFileCache.expandInputMiddleman(middlemanArtifact));
              }
            }, actionStartTime);

      return null;
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

  @Override
  protected void postEvent(Object event) {
    EventBus bus = eventBus.get();
    if (bus != null) {
      bus.post(event);
    }
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
      reporter.error(action.getOwner().getLocation(), message);
      recordExecutionError();
    }
  }

  /** Describe an action, for use in error messages. */
  private String describeAction(Action action) {
    if (action.getOutputs().isEmpty()) {
      return "run " + action.prettyPrint();
    } else if (action.getActionType().isMiddleman()) {
      return "build " + action.prettyPrint();
    } else {
      return "build file " + action.getPrimaryOutput().prettyPrint();
    }
  }

  @Override
  protected void updateCache(Action action, Token token, MetadataHandler metadataHandler)
      throws ActionExecutionException {
    try {
      actionCacheChecker.afterExecution(action, token, metadataHandler);
    } catch (IOException e) {
      // Skyframe does all the filesystem access needed during the checkOutputs call, and if that
      // call failed, the action cache shouldn't be updated. So an IOException is impossible here.
      throw new IllegalStateException(
          "failed to update action cache for " + action.prettyPrint()
          + ", but all outputs should already have been checked", e);
    }
  }

  @Override
  protected void dumpRecordedOutErr(Action action, FileOutErr outErrBuffer) {
    OutErr outErr = reporter.getOutErr();
    outErrBuffer.dumpOutAsLatin1(outErr.getOutputStream());
    outErrBuffer.dumpErrAsLatin1(outErr.getErrorStream());
  }

  public void reportActionExecutionFailure(Action action) {
    postEvent(new ActionFailedEvent(action));
  }

  @Override
  protected void scheduleAndExecuteAction(Action action,
      ActionExecutionContext actionExecutionContext)
  throws ActionExecutionException, InterruptedException {
    executeActionTask(action, actionExecutionContext);
  }

  /**
   * Currently this is a copy of what we do in legacy. The main implication is that we are
   * printing the error to the top level reporter instead of the action reporter. Because of that
   * Skyframe nodes do not know about the errors happening in the execution phase. Even if we
   * change in the future to log to the action reporter (that would be done in
   * ActionExecutionNodeBuilder.build() method when we get a ActionExecutionException),
   * we probably do not want to also store the StdErr output, so dumpRecordedOutErr() should
   * still be called here.
   */
  @Override
  protected boolean reportErrorIfNotAbortingMode(ActionExecutionException ex,
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
    public File getFileFromDigest(ByteString digest) throws IOException {
      File file = perActionCache.getFileFromDigest(digest);
      return file != null ? file : perBuildFileCache.getFileFromDigest(digest);
    }
  }
}
