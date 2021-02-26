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


import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.CompletionContext.PathResolverFactory;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.analysis.ConfiguredObjectValue;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.MissingFileArtifactValue;
import com.google.devtools.build.lib.skyframe.CompletionFunction.TopLevelActionLookupKey;
import com.google.devtools.build.lib.skyframe.MetadataConsumerForMetrics.FilesMetricConsumer;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/** CompletionFunction builds the artifactsToBuild collection of a {@link ConfiguredTarget}. */
public final class CompletionFunction<
        ValueT extends ConfiguredObjectValue,
        ResultT extends SkyValue,
        KeyT extends TopLevelActionLookupKey,
        FailureT>
    implements SkyFunction {

  /** A strategy for completing the build. */
  interface Completor<
      ValueT, ResultT extends SkyValue, KeyT extends TopLevelActionLookupKey, FailureT> {

    /**
     * Returns the options which determine the artifacts to build for the top-level targets.
     *
     * <p>For the Top level targets we made a conscious decision to include the
     * TopLevelArtifactContext within the SkyKey as an argument to the CompletionFunction rather
     * than a separate SkyKey. As a result we do have <num top level targets> extra SkyKeys for
     * every unique TopLevelArtifactContexts used over the lifetime of Blaze. This is a minor
     * tradeoff, since it significantly improves null build times when we're switching the
     * TopLevelArtifactContexts frequently (common for IDEs), by reusing existing SkyKeys from
     * earlier runs, instead of causing an eager invalidation were the TopLevelArtifactContext
     * modeled as a separate SkyKey.
     */

    /** Creates an event reporting an absent input artifact. */
    Event getRootCauseError(ValueT value, KeyT key, LabelCause rootCause, Environment env)
        throws InterruptedException;

    /** Creates an error message reporting {@code missingCount} missing input files. */
    MissingInputFileException getMissingFilesException(
        ValueT value, KeyT key, int missingCount, Environment env) throws InterruptedException;

    /** Provides a successful completion value. */
    ResultT getResult();

    /**
     * Creates supplementary data needed to call {@link #createFailed(Object, NestedSet,
     * CompletionContext, ImmutableMap, Object)}; returns null if skyframe found missing values.
     */
    @Nullable
    FailureT getFailureData(KeyT key, ValueT value, Environment env) throws InterruptedException;

    /** Creates a failed completion value. */
    ExtendedEventHandler.Postable createFailed(
        ValueT value,
        NestedSet<Cause> rootCauses,
        CompletionContext ctx,
        ImmutableMap<String, ArtifactsInOutputGroup> outputs,
        FailureT failureData)
        throws InterruptedException;

    /** Creates a succeeded completion value; returns null if skyframe found missing values. */
    @Nullable
    ExtendedEventHandler.Postable createSucceeded(
        KeyT skyKey,
        ValueT value,
        CompletionContext completionContext,
        ArtifactsToBuild artifactsToBuild,
        Environment env)
        throws InterruptedException;
  }

  interface TopLevelActionLookupKey extends SkyKey {
    ActionLookupKey actionLookupKey();

    TopLevelArtifactContext topLevelArtifactContext();
  }

  /**
   * Reduce an ArtifactsToBuild to only the Artifacts that were actually built (used when reporting
   * a failed target/aspect's completed outputs).
   */
  private static ImmutableMap<String, ArtifactsInOutputGroup>
      filterArtifactOutputGroupsToBuiltArtifacts(
          ImmutableSet<Artifact> builtArtifacts, ArtifactsToBuild allArtifactsToBuild) {
    return ImmutableMap.copyOf(
        Maps.filterValues(
            allArtifactsToBuild.getAllArtifactsByOutputGroup(),
            // Iterating over all artifacts in the output group although we already iterated over
            // the set while collecting all builtArtifacts. Ideally we would have a
            // NestedSetIntersectionView that would not require duplicating some-or-all of the
            // original NestedSet.
            artifactsInOutputGroup ->
                builtArtifacts.containsAll(artifactsInOutputGroup.getArtifacts().toList())));
  }

  private final PathResolverFactory pathResolverFactory;
  private final Completor<ValueT, ResultT, KeyT, FailureT> completor;
  private final SkyframeActionExecutor skyframeActionExecutor;
  private final FilesMetricConsumer topLevelArtifactsMetric;

  CompletionFunction(
      PathResolverFactory pathResolverFactory,
      Completor<ValueT, ResultT, KeyT, FailureT> completor,
      SkyframeActionExecutor skyframeActionExecutor,
      FilesMetricConsumer topLevelArtifactsMetric) {
    this.pathResolverFactory = pathResolverFactory;
    this.completor = completor;
    this.skyframeActionExecutor = skyframeActionExecutor;
    this.topLevelArtifactsMetric = topLevelArtifactsMetric;
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

    // Avoid iterating over nested set twice.
    ImmutableList<Artifact> allArtifacts = artifactsToBuild.getAllArtifacts().toList();
    Map<SkyKey, ValueOrException2<ActionExecutionException, IOException>> inputDeps =
        env.getValuesOrThrow(
            Artifact.keys(allArtifacts), ActionExecutionException.class, IOException.class);

    ActionInputMap inputMap = new ActionInputMap(inputDeps.size());
    Map<Artifact, ImmutableCollection<Artifact>> expandedArtifacts = new HashMap<>();
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets = new HashMap<>();
    Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts = new HashMap<>();
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets = new HashMap<>();

    int missingCount = 0;
    ActionExecutionException firstActionExecutionException = null;
    MissingInputFileException missingInputException = null;
    NestedSetBuilder<Cause> rootCausesBuilder = NestedSetBuilder.stableOrder();
    ImmutableSet.Builder<Artifact> builtArtifactsBuilder = ImmutableSet.builder();
    // Don't double-count files due to Skyframe restarts.
    FilesMetricConsumer currentConsumer = new FilesMetricConsumer();
    for (Artifact input : allArtifacts) {
      try {
        SkyValue artifactValue = inputDeps.get(Artifact.key(input)).get();
        if (artifactValue != null) {
          if (artifactValue instanceof MissingFileArtifactValue) {
            missingCount++;
            handleMissingFile(
                input,
                (MissingFileArtifactValue) artifactValue,
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
          }
        }
      } catch (ActionExecutionException e) {
        rootCausesBuilder.addTransitive(e.getRootCauses());
        // Prefer a catastrophic exception as the one we propagate.
        if (firstActionExecutionException == null
            || (!firstActionExecutionException.isCatastrophe() && e.isCatastrophe())) {
          firstActionExecutionException = e;
        }
      } catch (IOException e) {
        if (!input.isSourceArtifact()) {
          BugReport.sendBugReport(
              new IllegalStateException(
                  "Unexpected IOException for generated artifact: " + input, e));
        }
        missingCount++;
        handleMissingFile(
            input,
            ArtifactFunction.makeIOExceptionSourceInputFileValue(input, e),
            rootCausesBuilder,
            env,
            value,
            key);
      }
    }
    expandedFilesets.putAll(topLevelFilesets);

    if (missingCount > 0) {
      missingInputException = completor.getMissingFilesException(value, key, missingCount, env);
      if (missingInputException == null) {
        return null;
      }
    }

    NestedSet<Cause> rootCauses = rootCausesBuilder.build();
    @Nullable FailureT failureData = null;
    if (!rootCauses.isEmpty()) {
      failureData = completor.getFailureData(key, value, env);
      if (failureData == null) {
        return null;
      }
    }

    final CompletionContext ctx;
    try {
      ctx =
          CompletionContext.create(
              expandedArtifacts,
              archivedTreeArtifacts,
              expandedFilesets,
              key.topLevelArtifactContext().expandFilesets(),
              key.topLevelArtifactContext().fullyResolveFilesetSymlinks(),
              inputMap,
              pathResolverFactory,
              skyframeActionExecutor.getExecRoot(),
              workspaceNameValue.getName());
    } catch (IOException e) {
      throw new CompletionFunctionException(e);
    }

    if (!rootCauses.isEmpty()) {
      ImmutableMap<String, ArtifactsInOutputGroup> builtOutputs =
          filterArtifactOutputGroupsToBuiltArtifacts(
              builtArtifactsBuilder.build(), artifactsToBuild);
      env.getListener()
          .post(completor.createFailed(value, rootCauses, ctx, builtOutputs, failureData));
      if (firstActionExecutionException != null) {
        throw new CompletionFunctionException(firstActionExecutionException);
      } else {
        throw new CompletionFunctionException(missingInputException);
      }
    }

    // Only check for missing values *after* reporting errors: if there are missing files in a build
    // with --nokeep_going, there may be missing dependencies during error bubbling, we still need
    // to report the error.
    if (env.valuesMissing()) {
      return null;
    }

    ExtendedEventHandler.Postable postable =
        completor.createSucceeded(key, value, ctx, artifactsToBuild, env);
    if (postable == null) {
      return null;
    }
    env.getListener().post(postable);
    topLevelArtifactsMetric.mergeIn(currentConsumer);
    return completor.getResult();
  }

  private void handleMissingFile(
      Artifact input,
      MissingFileArtifactValue artifactValue,
      NestedSetBuilder<Cause> rootCausesBuilder,
      Environment env,
      ValueT value,
      KeyT key)
      throws InterruptedException {
    LabelCause cause =
        ActionExecutionFunction.createLabelCause(
            input, artifactValue, key.actionLookupKey().getLabel());
    rootCausesBuilder.add(cause);
    env.getListener().handle(completor.getRootCauseError(value, key, cause, env));
    skyframeActionExecutor.recordExecutionError();
  }

  @Nullable
  static <ValueT extends ConfiguredObjectValue>
      Pair<ValueT, ArtifactsToBuild> getValueAndArtifactsToBuild(
          TopLevelActionLookupKey key, Environment env) throws InterruptedException {
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

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((TopLevelActionLookupKey) skyKey).actionLookupKey().getLabel());
  }

  private static final class CompletionFunctionException extends SkyFunctionException {

    private final ActionExecutionException actionException;

    public CompletionFunctionException(ActionExecutionException e) {
      super(e, Transience.PERSISTENT);
      this.actionException = e;
    }

    public CompletionFunctionException(MissingInputFileException e) {
      super(e, Transience.TRANSIENT);
      this.actionException = null;
    }

    public CompletionFunctionException(IOException e) {
      super(e, Transience.TRANSIENT);
      this.actionException = null;
    }

    @Override
    public boolean isCatastrophic() {
      return actionException != null && actionException.isCatastrophe();
    }
  }
}
