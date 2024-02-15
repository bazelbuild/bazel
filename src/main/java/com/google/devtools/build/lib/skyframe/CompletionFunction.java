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
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.CompletionContext.PathResolverFactory;
import com.google.devtools.build.lib.actions.EventReportingArtifacts;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.ImportantOutputHandler;
import com.google.devtools.build.lib.actions.InputFileErrorException;
import com.google.devtools.build.lib.analysis.ConfiguredObjectValue;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.SuccessfulArtifactFilter;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.MissingArtifactValue;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.SourceArtifactException;
import com.google.devtools.build.lib.skyframe.MetadataConsumerForMetrics.FilesMetricConsumer;
import com.google.devtools.build.lib.skyframe.rewinding.ActionRewindStrategy;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/** CompletionFunction builds the artifactsToBuild collection of a {@link ConfiguredTarget}. */
public final class CompletionFunction<
        ValueT extends ConfiguredObjectValue,
        ResultT extends SkyValue,
        KeyT extends TopLevelActionLookupKeyWrapper,
        FailureT>
    implements SkyFunction {

  /** A strategy for completing the build. */
  interface Completor<
      ValueT, ResultT extends SkyValue, KeyT extends TopLevelActionLookupKeyWrapper, FailureT> {

    /** Creates an event reporting an absent input artifact. */
    Event getRootCauseError(ValueT value, KeyT key, LabelCause rootCause, Environment env)
        throws InterruptedException;

    @Nullable
    Object getLocationIdentifier(ValueT value, KeyT key, Environment env)
        throws InterruptedException;

    /** Provides a successful completion value. */
    ResultT getResult();

    /**
     * Creates supplementary data needed to call {@link #createFailed}; returns null if skyframe
     * found missing values.
     */
    @Nullable
    FailureT getFailureData(KeyT key, ValueT value, Environment env) throws InterruptedException;

    /**
     * Creates a failed completion event.
     *
     * <p>The event must be {@linkplain Postable#storeForReplay stored}.
     */
    Postable createFailed(
        KeyT skyKey,
        NestedSet<Cause> rootCauses,
        CompletionContext ctx,
        ImmutableMap<String, ArtifactsInOutputGroup> outputs,
        FailureT failureData)
        throws InterruptedException;

    /**
     * Creates a succeeded completion event; returns null if skyframe found missing values.
     *
     * <p>The event must be {@linkplain Postable#storeForReplay stored}.
     */
    @Nullable
    EventReportingArtifacts createSucceeded(
        KeyT skyKey,
        ValueT value,
        CompletionContext completionContext,
        ArtifactsToBuild artifactsToBuild,
        Environment env)
        throws InterruptedException;
  }

  private final PathResolverFactory pathResolverFactory;
  private final Completor<ValueT, ResultT, KeyT, FailureT> completor;
  private final SkyframeActionExecutor skyframeActionExecutor;
  private final FilesMetricConsumer topLevelArtifactsMetric;
  private final ActionRewindStrategy actionRewindStrategy;
  private final BugReporter bugReporter;

  CompletionFunction(
      PathResolverFactory pathResolverFactory,
      Completor<ValueT, ResultT, KeyT, FailureT> completor,
      SkyframeActionExecutor skyframeActionExecutor,
      FilesMetricConsumer topLevelArtifactsMetric,
      ActionRewindStrategy actionRewindStrategy,
      BugReporter bugReporter) {
    this.pathResolverFactory = checkNotNull(pathResolverFactory);
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
    WorkspaceNameValue workspaceNameValue =
        (WorkspaceNameValue) env.getValue(WorkspaceNameValue.key());
    if (workspaceNameValue == null) {
      return null;
    }

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

    ActionInputMap inputMap = new ActionInputMap(bugReporter, allArtifacts.size());
    // Prepare an ActionInputMap for important artifacts separately, to be used by BEP events. The
    // _validation output group can contain orders of magnitude more unimportant artifacts than
    // there are important artifacts, and BEP events will retain the ActionInputMap until the
    // event is delivered to transports. If the BEP events reference *all* artifacts it can increase
    // heap high-watermark by multiple GB.
    ActionInputMap importantInputMap;
    ImmutableCollection<Artifact> importantArtifacts;
    if (allArtifactsAreImportant) {
      importantArtifacts = allArtifacts;
      importantInputMap = inputMap;
    } else {
      importantArtifacts = artifactsToBuild.getImportantArtifacts().toSet();
      importantInputMap = new ActionInputMap(bugReporter, importantArtifacts.size());
    }

    Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts = new HashMap<>();
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets = new HashMap<>();
    Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts = new HashMap<>();
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets = new HashMap<>();

    ActionExecutionException firstActionExecutionException = null;
    NestedSetBuilder<Cause> rootCausesBuilder = NestedSetBuilder.stableOrder();
    ImmutableSet.Builder<Artifact> builtArtifactsBuilder = ImmutableSet.builder();
    // Don't double-count files due to Skyframe restarts.
    FilesMetricConsumer currentConsumer = new FilesMetricConsumer();
    for (Artifact input : allArtifacts) {
      try {
        SkyValue artifactValue =
            inputDeps.getOrThrow(
                Artifact.key(input), ActionExecutionException.class, SourceArtifactException.class);
        if (artifactValue != null) {
          if (artifactValue instanceof MissingArtifactValue) {
            handleSourceFileError(
                input,
                ((MissingArtifactValue) artifactValue).getDetailedExitCode(),
                rootCausesBuilder,
                env,
                value,
                key);
          } else {
            builtArtifactsBuilder.add(input);
            ActionInputMapHelper.addToMap(
                inputMap,
                expandedArtifacts,
                archivedTreeArtifacts,
                expandedFilesets,
                topLevelFilesets,
                input,
                artifactValue,
                env,
                currentConsumer);
            if (!allArtifactsAreImportant && importantArtifacts.contains(input)) {
              // Calling #addToMap a second time with `input` and `artifactValue` will perform no-op
              // updates to the secondary collections passed in (eg. expandedArtifacts,
              // topLevelFilesets). MetadataConsumerForMetrics.NO_OP is used to avoid
              // double-counting.
              ActionInputMapHelper.addToMap(
                  importantInputMap,
                  expandedArtifacts,
                  archivedTreeArtifacts,
                  expandedFilesets,
                  topLevelFilesets,
                  input,
                  artifactValue,
                  env);
            }
          }
        }
      } catch (ActionExecutionException e) {
        rootCausesBuilder.addTransitive(e.getRootCauses());
        // Prefer a catastrophic exception as the one we propagate.
        if (firstActionExecutionException == null
            || (!firstActionExecutionException.isCatastrophe() && e.isCatastrophe())) {
          firstActionExecutionException = e;
        }
      } catch (SourceArtifactException e) {
        if (!input.isSourceArtifact()) {
          bugReporter.logUnexpected(
              e, "Non-source artifact had SourceArtifactException: %s", input);
        }
        handleSourceFileError(input, e.getDetailedExitCode(), rootCausesBuilder, env, value, key);
      }
    }
    expandedFilesets.putAll(topLevelFilesets);

    NestedSet<Cause> rootCauses = rootCausesBuilder.build();
    @Nullable FailureT failureData = null;
    if (!rootCauses.isEmpty()) {
      failureData = completor.getFailureData(key, value, env);
      if (failureData == null) {
        return null;
      }
    }

    CompletionContext ctx =
        CompletionContext.create(
            expandedArtifacts,
            expandedFilesets,
            key.topLevelArtifactContext().expandFilesets(),
            key.topLevelArtifactContext().fullyResolveFilesetSymlinks(),
            inputMap,
            importantInputMap,
            pathResolverFactory,
            skyframeActionExecutor.getExecRoot(),
            workspaceNameValue.getName());

    if (!rootCauses.isEmpty()) {
      RewindPlan rewindPlan = null;
      ImmutableSet<Artifact> builtArtifacts = builtArtifactsBuilder.build();
      if (!builtArtifacts.isEmpty()) {
        rewindPlan =
            informImportantOutputHandler(
                key,
                env,
                allArtifactsAreImportant
                    ? builtArtifacts
                    : Sets.intersection(builtArtifacts, (Set<?>) importantArtifacts),
                ctx);
        if (rewindPlan != null) {
          // Filter out lost outputs from the set of built artifacts so that they are not reported.
          // If rewinding is successful, we'll report them later on.
          builtArtifacts =
              Sets.difference(builtArtifacts, new HashSet<>(rewindPlan.lostOutputs))
                  .immutableCopy();
        }
      }
      ImmutableMap<String, ArtifactsInOutputGroup> builtOutputs =
          new SuccessfulArtifactFilter(builtArtifacts)
              .filterArtifactsInOutputGroup(artifactsToBuild.getAllArtifactsByOutputGroup());
      Postable event = completor.createFailed(key, rootCauses, ctx, builtOutputs, failureData);
      checkStored(event, key);
      env.getListener().post(event);
      if (rewindPlan != null) {
        // Only return a reset after posting the failed event. If we're in --nokeep_going mode, the
        // attempt to rewind will be ignored, so this is our only opportunity to post the event. If
        // we're in --keep_going mode, rewinding will take place, the event won't actually get
        // emitted (per the spec of SkyFunction.Environment#getListener for stored events), and
        // we'll get another opportunity to post an event after rewinding.
        return rewindPlan.reset;
      }
      if (firstActionExecutionException != null) {
        throw new CompletionFunctionException(firstActionExecutionException);
      }
      // locationPrefix theoretically *could* be null because of missing deps, but not in reality,
      // and we're not allowed to wait for deps to be ready if we're failing anyway.
      @Nullable Object locationPrefix = completor.getLocationIdentifier(value, key, env);
      Pair<DetailedExitCode, String> codeAndMessage =
          ActionExecutionFunction.createSourceErrorCodeAndMessage(rootCauses.toList(), key);
      String message;
      if (locationPrefix instanceof Location) {
        message = codeAndMessage.getSecond();
        env.getListener().handle(Event.error((Location) locationPrefix, message));
      } else {
        message = (locationPrefix == null ? "" : locationPrefix + " ") + codeAndMessage.getSecond();
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

    RewindPlan rewindPlan = informImportantOutputHandler(key, env, importantArtifacts, ctx);
    if (rewindPlan != null) {
      return rewindPlan.reset; // Initiate action rewinding to regenerate lost outputs.
    }

    Postable event = completor.createSucceeded(key, value, ctx, artifactsToBuild, env);
    if (event == null) {
      return null;
    }
    checkStored(event, key);
    env.getListener().post(event);
    topLevelArtifactsMetric.mergeIn(currentConsumer);

    return completor.getResult();
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
    env.getListener().handle(completor.getRootCauseError(value, key, cause, env));
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
   * Calls {@link ImportantOutputHandler#processAndGetLostArtifacts}.
   *
   * <p>If any outputs are lost, returns a {@link RewindPlan} containing a {@link Reset} which used
   * to initiate action rewinding and regenerate the lost outputs. Otherwise, returns {@code null}.
   */
  @Nullable
  private RewindPlan informImportantOutputHandler(
      KeyT key, Environment env, Collection<Artifact> importantArtifacts, CompletionContext ctx)
      throws InterruptedException {
    ImportantOutputHandler importantOutputHandler =
        skyframeActionExecutor.getActionContextRegistry().getContext(ImportantOutputHandler.class);
    if (importantOutputHandler == ImportantOutputHandler.NO_OP) {
      return null; // Avoid expanding artifacts if the default no-op handler is installed.
    }

    ImmutableMap<String, ActionInput> lostOutputs =
        importantOutputHandler.processAndGetLostArtifacts(
            ctx.expand(importantArtifacts),
            new ActionInputMetadataProvider(
                skyframeActionExecutor.getExecRoot().asFragment(),
                ctx.getImportantInputMap(),
                ctx.getExpandedFilesets()));
    if (lostOutputs.isEmpty()) {
      return null;
    }

    Reset reset =
        actionRewindStrategy.prepareRewindPlanForLostTopLevelOutputs(
            key,
            ImmutableSet.copyOf(Artifact.keys(importantArtifacts)),
            lostOutputs,
            ctx.getDepOwners(lostOutputs.values()),
            env);
    return new RewindPlan(lostOutputs.values(), reset);
  }

  private static void checkStored(Postable event, TopLevelActionLookupKeyWrapper key) {
    checkState(
        event.storeForReplay(), "Completion events must be stored, got %s for %s", event, key);
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((TopLevelActionLookupKeyWrapper) skyKey).actionLookupKey().getLabel());
  }

  /**
   * Wraps a collection of lost outputs and a {@link Reset} to initiate rewinding and regenerate
   * them.
   */
  private static final class RewindPlan {
    private final ImmutableCollection<ActionInput> lostOutputs;
    private final Reset reset;

    private RewindPlan(ImmutableCollection<ActionInput> lostOutputs, Reset reset) {
      this.lostOutputs = lostOutputs;
      this.reset = reset;
    }
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

    @Override
    public boolean isCatastrophic() {
      return actionException != null && actionException.isCatastrophe();
    }
  }
}
