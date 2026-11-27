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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.CompletionContext.PathResolverFactory;
import com.google.devtools.build.lib.actions.ImportantOutputHandler;
import com.google.devtools.build.lib.actions.ImportantOutputHandler.ImportantOutputException;
import com.google.devtools.build.lib.actions.ImportantOutputHandler.LostArtifacts;
import com.google.devtools.build.lib.actions.InputFileErrorException;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.TopLevelOutputException;
import com.google.devtools.build.lib.analysis.ConfiguredObjectValue;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.SuccessfulArtifactFilter;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.MissingArtifactValue;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.SourceArtifactException;
import com.google.devtools.build.lib.skyframe.MetadataConsumerForMetrics.FilesMetricConsumer;
import com.google.devtools.build.lib.skyframe.rewinding.ActionRewindException;
import com.google.devtools.build.lib.skyframe.rewinding.ActionRewindStrategy;
import com.google.devtools.build.lib.skyframe.rewinding.ActionRewindStrategy.RewindPlanResult;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/** CompletionFunction builds the artifactsToBuild collection of a {@link ConfiguredTarget}. */
public final class CompletionFunction<
        ValueT extends ConfiguredObjectValue,
        ResultT extends SkyValue,
        KeyT extends TopLevelActionLookupKeyWrapper>
    implements SkyFunction {

  /**
   * A strategy for completing the build.
   *
   * <p>Any Skyframe lookups in methods passed an {@link Environment} must return an already-done
   * value. For example, it is acceptable to call {@link
   * ConfiguredTargetAndData#fromExistingConfiguredTargetInSkyframe}.
   */
  interface Completor<
      ValueT, ResultT extends SkyValue, KeyT extends TopLevelActionLookupKeyWrapper> {

    /** Creates an event reporting an absent input artifact. */
    Event getRootCauseError(KeyT key, ValueT value, LabelCause rootCause, Environment env)
        throws InterruptedException;

    Object getLocationIdentifier(KeyT key, ValueT value, Environment env)
        throws InterruptedException;

    /** Provides a successful completion value. */
    ResultT getResult();

    /** Creates a factory for the failed completion event. */
    EventFactory createFailed(
        KeyT skyKey,
        ValueT value,
        NestedSet<Cause> rootCauses,
        ImmutableMap<String, ArtifactsInOutputGroup> outputs,
        Environment env)
        throws InterruptedException;

    /** Creates a factory for the succeeded completion event. */
    EventFactory createSucceeded(
        KeyT skyKey, ValueT value, ArtifactsToBuild artifactsToBuild, Environment env)
        throws InterruptedException;
  }

  /**
   * Creates a completion event from the {@link CompletionContext} to report it with.
   *
   * <p>The factory is stored in the Skyframe graph in place of the event, so that the graph does
   * not retain the context. It must not retain anything expensive either.
   */
  interface EventFactory {
    Postable create(CompletionContext ctx);
  }

  /**
   * Returns the last known value of a node in the Skyframe graph, or null if it has none.
   *
   * <p>Only used for nodes that the completion node depends on, to recreate the {@link
   * CompletionContext} of a completion event that is replayed.
   */
  interface ArtifactValueLookup {
    @Nullable
    SkyValue getLastKnownValue(SkyKey key);
  }

  private final PathResolverFactory pathResolverFactory;
  private final ArtifactValueLookup artifactValueLookup;
  private final Completor<ValueT, ResultT, KeyT> completor;
  private final SkyframeActionExecutor skyframeActionExecutor;
  private final FilesMetricConsumer topLevelArtifactsMetric;
  private final ActionRewindStrategy actionRewindStrategy;
  private final BugReporter bugReporter;

  CompletionFunction(
      PathResolverFactory pathResolverFactory,
      ArtifactValueLookup artifactValueLookup,
      Completor<ValueT, ResultT, KeyT> completor,
      SkyframeActionExecutor skyframeActionExecutor,
      FilesMetricConsumer topLevelArtifactsMetric,
      ActionRewindStrategy actionRewindStrategy,
      BugReporter bugReporter) {
    this.pathResolverFactory = checkNotNull(pathResolverFactory);
    this.artifactValueLookup = checkNotNull(artifactValueLookup);
    this.completor = checkNotNull(completor);
    this.skyframeActionExecutor = checkNotNull(skyframeActionExecutor);
    this.topLevelArtifactsMetric = checkNotNull(topLevelArtifactsMetric);
    this.actionRewindStrategy = checkNotNull(actionRewindStrategy);
    this.bugReporter = checkNotNull(bugReporter);
  }

  @SuppressWarnings("unchecked") // Cast to KeyT
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws CompletionFunctionException, InterruptedException {
    KeyT key = (KeyT) skyKey;
    Pair<ValueT, ArtifactsToBuild> valueAndArtifactsToBuild = getValueAndArtifactsToBuild(key, env);
    if (env.valuesMissing()) {
      return null;
    }
    ValueT value = valueAndArtifactsToBuild.first;
    ArtifactsToBuild artifactsToBuild = valueAndArtifactsToBuild.second;

    ImmutableList<Artifact> allArtifacts = artifactsToBuild.getAllArtifacts().toList();
    SkyframeLookupResult inputDeps = env.getValuesAndExceptions(Artifact.keys(allArtifacts));

    boolean allArtifactsAreImportant = artifactsToBuild.areAllOutputGroupsImportant();

    ActionInputMap inputMap = new ActionInputMap(allArtifacts.size());
    // Prepare an ActionInputMap for important artifacts separately, to be used by BEP events. The
    // _validation output group can contain orders of magnitude more unimportant artifacts than
    // there are important artifacts, and BEP events will retain the ActionInputMap until the
    // event is delivered to transports. If the BEP events reference *all* artifacts it can increase
    // heap high-watermark by multiple GB. (The Skyframe graph does not retain the map: see
    // StoredCompletionEvent.)
    ActionInputMap importantInputMap;
    ImmutableCollection<Artifact> importantArtifacts;
    if (allArtifactsAreImportant) {
      importantArtifacts = allArtifacts;
      importantInputMap = inputMap;
    } else {
      importantArtifacts = artifactsToBuild.getImportantArtifacts().toSet();
      importantInputMap = new ActionInputMap(importantArtifacts.size());
    }

    ActionExecutionException worstActionExecutionException = null;
    NestedSetBuilder<Cause> rootCausesBuilder = NestedSetBuilder.stableOrder();
    Set<Artifact> builtArtifacts = new HashSet<>();
    // Don't double-count files due to Skyframe restarts.
    FilesMetricConsumer currentConsumer = new FilesMetricConsumer();
    for (Artifact input : allArtifacts) {
      try {
        SkyValue artifactValue =
            inputDeps.getOrThrow(
                Artifact.key(input), ActionExecutionException.class, SourceArtifactException.class);
        if (artifactValue == null) {
          continue;
        }
        if (artifactValue instanceof MissingArtifactValue) {
          handleSourceFileError(
              input,
              ((MissingArtifactValue) artifactValue).getDetailedExitCode(),
              rootCausesBuilder,
              env,
              value,
              key);
        } else {
          builtArtifacts.add(input);
          ActionInputMapHelper.addToMap(inputMap, input, artifactValue, currentConsumer);
          if (!allArtifactsAreImportant && importantArtifacts.contains(input)) {
            // Calling #addToMap a second time with `input` and `artifactValue` will perform no-op
            // updates to the secondary collections passed in (eg. treeArtifacts, expandedFilesets).
            // MetadataConsumerForMetrics.NO_OP is used to avoid double-counting.
            ActionInputMapHelper.addToMap(
                importantInputMap, input, artifactValue, MetadataConsumerForMetrics.NO_OP);
          }
        }
      } catch (ActionExecutionException e) {
        if (e.getRootCauses().isEmpty()) {
          BugReport.sendNonFatalBugReport(
              new IllegalStateException(
                  "Caught ActionExecutionException from %s with no root causes".formatted(input),
                  e));
        } else {
          rootCausesBuilder.addTransitive(e.getRootCauses());
        }
        // Prefer a catastrophic exception as the one we propagate.
        if (worstActionExecutionException == null) {
          worstActionExecutionException = e;
        } else {
          worstActionExecutionException =
              ActionExecutionException.SEVERITY_ORDERING.max(worstActionExecutionException, e);
        }
      } catch (SourceArtifactException e) {
        if (!input.isSourceArtifact()) {
          bugReporter.logUnexpected(
              e, "Non-source artifact had SourceArtifactException: %s", input);
        }
        handleSourceFileError(input, e.getDetailedExitCode(), rootCausesBuilder, env, value, key);
      }
    }
    NestedSet<Cause> rootCauses = rootCausesBuilder.build();
    if (!rootCauses.isEmpty()) {
      RewindPlanResult rewindPlanResult = null;
      if (!builtArtifacts.isEmpty()) {
        // In error bubbling, we may be interrupted by Skyframe. Ensure that the interrupt doesn't
        // prevent us from staging built artifacts and posting the failed event.
        boolean interruptedDuringErrorBubbling = env.inErrorBubbling() && Thread.interrupted();
        try {
          rewindPlanResult =
              informImportantOutputHandler(
                  key,
                  value,
                  env,
                  ImmutableList.copyOf(
                      allArtifactsAreImportant
                          ? builtArtifacts
                          : Iterables.filter(builtArtifacts, importantArtifacts::contains)),
                  rootCauses,
                  importantInputMap,
                  artifactsToBuild,
                  builtArtifacts,
                  inputMap);
        } finally {
          if (interruptedDuringErrorBubbling) {
            Thread.currentThread().interrupt();
          }
        }
      }
      postFailedEvent(
          key, value, rootCauses, importantInputMap, artifactsToBuild, builtArtifacts, env);
      if (rewindPlanResult != null) {
        // Only return a reset after posting the failed event. If we're in --nokeep_going mode, the
        // attempt to rewind will be ignored, so this is our only opportunity to post the event. If
        // we're in --keep_going mode, rewinding will take place, the event won't actually get
        // emitted (per the spec of SkyFunction.Environment#getListener for stored events), and
        // we'll get another opportunity to post an event after rewinding.
        return rewindPlanResult.toNullIfMissingDependenciesElseReset();
      }
      if (worstActionExecutionException != null) {
        throw new CompletionFunctionException(worstActionExecutionException);
      }
      Object locationPrefix = completor.getLocationIdentifier(key, value, env);
      Pair<DetailedExitCode, String> codeAndMessage =
          ActionExecutionFunction.createSourceErrorCodeAndMessage(rootCauses.toList(), key);
      String message;
      if (locationPrefix instanceof Location) {
        message = codeAndMessage.getSecond();
        env.getListener().handle(Event.error((Location) locationPrefix, message));
      } else {
        message = locationPrefix + " " + codeAndMessage.getSecond();
        env.getListener().handle(Event.error(message));
      }
      throw new CompletionFunctionException(
          new InputFileErrorException(message, codeAndMessage.getFirst()));
    }

    // Only check for missing values *after* reporting errors: if there are missing files in a build
    // with --nokeep_going, there may be missing dependencies during error bubbling, we still need
    // to report the error.
    if (env.valuesMissing()) {
      return null;
    }

    RewindPlanResult rewindPlanResult =
        informImportantOutputHandler(
            key,
            value,
            env,
            importantArtifacts,
            rootCauses,
            importantInputMap,
            artifactsToBuild,
            builtArtifacts,
            inputMap);
    if (rewindPlanResult != null) {
      // Either initiates action rewinding to generate lost inputs or requests a Skyframe restart to
      // wait for missing analysis dependencies.
      return rewindPlanResult.toNullIfMissingDependenciesElseReset();
    }

    env.getListener()
        .post(
            new StoredCompletionEvent(
                completor.createSucceeded(key, value, artifactsToBuild, env),
                artifactsToBuild,
                importantInputMap));
    topLevelArtifactsMetric.mergeIn(currentConsumer);

    return completor.getResult();
  }

  private void postFailedEvent(
      KeyT key,
      ValueT value,
      NestedSet<Cause> rootCauses,
      ActionInputMap importantInputMap,
      ArtifactsToBuild artifactsToBuild,
      Set<Artifact> builtArtifacts,
      Environment env)
      throws InterruptedException {
    ImmutableMap<String, ArtifactsInOutputGroup> builtOutputs =
        new SuccessfulArtifactFilter(ImmutableSet.copyOf(builtArtifacts))
            .filterArtifactsInOutputGroup(artifactsToBuild.getAllArtifactsByOutputGroup());
    env.getListener()
        .post(
            new StoredCompletionEvent(
                completor.createFailed(key, value, rootCauses, builtOutputs, env),
                artifactsToBuild,
                importantInputMap));
  }

  private void handleSourceFileError(
      Artifact input,
      DetailedExitCode detailedExitCode,
      NestedSetBuilder<Cause> rootCausesBuilder,
      Environment env,
      ValueT value,
      KeyT key)
      throws InterruptedException {
    LabelCause cause =
        ActionExecutionFunction.createLabelCause(
            input, detailedExitCode, key.actionLookupKey().getLabel(), bugReporter);
    rootCausesBuilder.add(cause);
    env.getListener().handle(completor.getRootCauseError(key, value, cause, env));
    skyframeActionExecutor.recordExecutionError();
  }

  @Nullable
  static <ValueT extends ConfiguredObjectValue>
      Pair<ValueT, ArtifactsToBuild> getValueAndArtifactsToBuild(
          TopLevelActionLookupKeyWrapper key, Environment env) throws InterruptedException {
    @SuppressWarnings("unchecked")
    ValueT value = (ValueT) env.getValue(key.actionLookupKey());
    if (env.valuesMissing()) {
      return null;
    }

    TopLevelArtifactContext topLevelContext = key.topLevelArtifactContext();
    ArtifactsToBuild artifactsToBuild =
        TopLevelArtifactHelper.getAllArtifactsToBuild(value.getConfiguredObject(), topLevelContext);
    return Pair.of(value, artifactsToBuild);
  }

  /**
   * Calls {@link ImportantOutputHandler#processOutputsAndGetLostArtifacts}.
   *
   * <p>If any outputs are lost, returns a {@link Reset} which can be used to initiate action
   * rewinding and regenerate the lost outputs. Otherwise, returns {@code null}.
   */
  @Nullable
  private RewindPlanResult informImportantOutputHandler(
      KeyT key,
      ValueT value,
      Environment env,
      ImmutableCollection<Artifact> importantArtifacts,
      NestedSet<Cause> rootCauses,
      ActionInputMap importantInputMap,
      ArtifactsToBuild artifactsToBuild,
      Set<Artifact> builtArtifacts,
      ActionInputMap inputMap)
      throws CompletionFunctionException, InterruptedException {
    var importantOutputHandler =
        skyframeActionExecutor.getActionContextRegistry().getContext(ImportantOutputHandler.class);
    if (importantOutputHandler == null) {
      return null;
    }

    Label label = key.actionLookupKey().getLabel();
    InputMetadataProvider metadataProvider =
        new ActionInputMetadataProvider(
            importantOutputHandler.requiresHiddenOutputMetadata() ? inputMap : importantInputMap);
    try {
      LostArtifacts lostOutputs;
      try (var ignored =
          GoogleAutoProfilerUtils.profiledAndLogged(
              "Informing important output handler of top-level outputs for " + label,
              ProfilerTask.INFO,
              ImportantOutputHandler.LOG_THRESHOLD)) {
        lostOutputs =
            importantOutputHandler.processOutputsAndGetLostArtifacts(
                importantArtifacts, metadataProvider);
      }
      if (lostOutputs.isEmpty()) {
        return null;
      }

      Iterable<Artifact> artifactsRelevantForRewinding = importantArtifacts;
      if (importantOutputHandler.requiresHiddenOutputMetadata()) {
        var hiddenTopLevelArtifacts =
            artifactsToBuild.getAllArtifactsByOutputGroup().get(OutputGroupInfo.HIDDEN_TOP_LEVEL);
        if (hiddenTopLevelArtifacts != null) {
          artifactsRelevantForRewinding =
              Iterables.concat(
                  artifactsRelevantForRewinding, hiddenTopLevelArtifacts.getArtifacts().toList());
        }
      }

      return actionRewindStrategy.prepareRewindPlanForLostTopLevelOutputs(
          key,
          ImmutableSet.copyOf(Artifact.keys(artifactsRelevantForRewinding)),
          lostOutputs.byDigest(),
          metadataProvider,
          builtArtifacts,
          env);
    } catch (ActionRewindException | ImportantOutputException e) {
      LabelCause cause = new LabelCause(label, e.getDetailedExitCode());
      rootCauses = NestedSetBuilder.fromNestedSet(rootCauses).add(cause).build();
      env.getListener().handle(completor.getRootCauseError(key, value, cause, env));
      skyframeActionExecutor.recordExecutionError();
      postFailedEvent(
          key, value, rootCauses, importantInputMap, artifactsToBuild, builtArtifacts, env);
      throw new CompletionFunctionException(
          new TopLevelOutputException(e.getMessage(), e.getDetailedExitCode()));
    }
  }

  /**
   * What the Skyframe graph stores for a completion event, to replay it when a later build finds
   * the completion node up to date.
   *
   * <p>A completion event has a {@link CompletionContext}, with the metadata of every important
   * output of the target or aspect and, depending on the output service, a file system made for
   * that metadata. That is too much to keep for every top-level target and aspect for as long as
   * the graph lives, and the metadata is in the graph already, in the nodes that the completion
   * node depends on. So this is stored in place of the event, and makes the event, with a context
   * of its own, each time it is reported. The event is then retained only by those it was posted
   * to, for as long as they need it.
   */
  private final class StoredCompletionEvent implements Postable {
    private final EventFactory eventFactory;
    private final ArtifactsToBuild artifactsToBuild;

    /**
     * The metadata that {@link #compute} collected, for the first time this is reported, which is
     * when the completion node is done. Null after that.
     */
    @Nullable private volatile ActionInputMap computedImportantInputMap;

    StoredCompletionEvent(
        EventFactory eventFactory,
        ArtifactsToBuild artifactsToBuild,
        ActionInputMap computedImportantInputMap) {
      this.eventFactory = checkNotNull(eventFactory);
      this.artifactsToBuild = checkNotNull(artifactsToBuild);
      this.computedImportantInputMap = checkNotNull(computedImportantInputMap);
    }

    @Override
    public void reportTo(ExtendedEventHandler handler) {
      ActionInputMap importantInputMap = computedImportantInputMap;
      if (importantInputMap != null) {
        computedImportantInputMap = null;
      } else {
        importantInputMap = recreateImportantInputMap();
      }
      handler.post(
          eventFactory.create(CompletionContext.create(importantInputMap, pathResolverFactory)));
    }

    /**
     * Collects the metadata of the important artifacts that were built, as {@link #compute} did,
     * from the nodes that the completion node depends on.
     */
    private ActionInputMap recreateImportantInputMap() {
      ImmutableList<Artifact> importantArtifacts =
          artifactsToBuild.getImportantArtifacts().toList();
      ActionInputMap importantInputMap = new ActionInputMap(importantArtifacts.size());
      for (Artifact artifact : importantArtifacts) {
        SkyValue artifactValue = artifactValueLookup.getLastKnownValue(Artifact.key(artifact));
        // An artifact without a value failed to build, and is not in the event either.
        if (artifactValue != null && !(artifactValue instanceof MissingArtifactValue)) {
          ActionInputMapHelper.addToMap(
              importantInputMap, artifact, artifactValue, MetadataConsumerForMetrics.NO_OP);
        }
      }
      return importantInputMap;
    }

    @Override
    public boolean storeForReplay() {
      return true;
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((TopLevelActionLookupKeyWrapper) skyKey).actionLookupKey().getLabel());
  }

  private static final class CompletionFunctionException extends SkyFunctionException {
    private final ActionExecutionException actionException;

    CompletionFunctionException(ActionExecutionException e) {
      super(e, Transience.PERSISTENT);
      this.actionException = e;
    }

    CompletionFunctionException(InputFileErrorException e) {
      // Not transient from the point of view of this SkyFunction.
      super(e, Transience.PERSISTENT);
      this.actionException = null;
    }

    CompletionFunctionException(TopLevelOutputException e) {
      super(e, Transience.TRANSIENT);
      this.actionException = null;
    }

    @Override
    public boolean isCatastrophic() {
      return actionException != null && actionException.isCatastrophe();
    }
  }
}
