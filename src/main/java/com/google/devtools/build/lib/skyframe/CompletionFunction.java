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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.CompletionContext.PathResolverFactory;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.MissingFileArtifactValue;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** CompletionFunction builds the artifactsToBuild collection of a {@link ConfiguredTarget}. */
public final class CompletionFunction<
        ValueT extends ConfiguredObjectValue, ResultT extends SkyValue>
    implements SkyFunction {

  /** A strategy for completing the build. */
  interface Completor<ValueT, ResultT extends SkyValue> {

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
    Event getRootCauseError(ValueT value, Cause rootCause, Environment env)
        throws InterruptedException;

    /** Creates an error message reporting {@code missingCount} missing input files. */
    MissingInputFileException getMissingFilesException(
        ValueT value, int missingCount, Environment env) throws InterruptedException;

    /** Provides a successful completion value. */
    ResultT getResult();

    /** Creates a failed completion value. */
    ExtendedEventHandler.Postable createFailed(
        ValueT value,
        NestedSet<Cause> rootCauses,
        NestedSet<ArtifactsInOutputGroup> outputs,
        Environment env,
        TopLevelArtifactContext topLevelArtifactContext)
        throws InterruptedException;

    /** Creates a succeeded completion value. */
    ExtendedEventHandler.Postable createSucceeded(
        SkyKey skyKey,
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
  private static NestedSet<ArtifactsInOutputGroup> filterArtifactOutputGroupsToBuiltArtifacts(
      ImmutableSet<Artifact> builtArtifacts, ArtifactsToBuild allArtifactsToBuild) {
    NestedSetBuilder<ArtifactsInOutputGroup> outputs = NestedSetBuilder.stableOrder();
    allArtifactsToBuild.getAllArtifactsByOutputGroup().toList().stream()
        .map(aog -> outputGroupIfAllArtifactsBuilt(aog, builtArtifacts))
        .flatMap(Streams::stream)
        .forEach(outputs::add);
    return outputs.build();
  }

  /**
   * Returns the given ArtifactsInOutputGroup unmodified if all referenced artifacts were
   * successfully built, and otherwise returns an empty Optional.
   */
  public static Optional<ArtifactsInOutputGroup> outputGroupIfAllArtifactsBuilt(
      ArtifactsInOutputGroup aog, ImmutableSet<Artifact> builtArtifacts) {
    // Iterating over all artifacts in the output group although we already iterated over the set
    // while collecting all builtArtifacts. Ideally we would have a NestedSetIntersectionView that
    // would not require duplicating some-or-all of the original NestedSet.
    if (aog.getArtifacts().toList().stream().allMatch(builtArtifacts::contains)) {
      return Optional.of(aog);
    }
    return Optional.empty();
  }

  private final PathResolverFactory pathResolverFactory;
  private final Completor<ValueT, ResultT> completor;
  private final Supplier<Path> execRootSupplier;

  CompletionFunction(
      PathResolverFactory pathResolverFactory,
      Completor<ValueT, ResultT> completor,
      Supplier<Path> execRootSupplier) {
    this.pathResolverFactory = pathResolverFactory;
    this.completor = completor;
    this.execRootSupplier = execRootSupplier;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws CompletionFunctionException, InterruptedException {
    WorkspaceNameValue workspaceNameValue =
        (WorkspaceNameValue) env.getValue(WorkspaceNameValue.key());
    if (workspaceNameValue == null) {
      return null;
    }

    TopLevelActionLookupKey key = (TopLevelActionLookupKey) skyKey;
    Pair<ValueT, ArtifactsToBuild> valueAndArtifactsToBuild = getValueAndArtifactsToBuild(key, env);
    if (env.valuesMissing()) {
      return null;
    }
    ValueT value = valueAndArtifactsToBuild.first;
    ArtifactsToBuild artifactsToBuild = valueAndArtifactsToBuild.second;

    // Avoid iterating over nested set twice.
    ImmutableList<Artifact> allArtifacts = artifactsToBuild.getAllArtifacts().toList();
    Map<SkyKey, ValueOrException<ActionExecutionException>> inputDeps =
        env.getValuesOrThrow(Artifact.keys(allArtifacts), ActionExecutionException.class);

    ActionInputMap inputMap = new ActionInputMap(inputDeps.size());
    Map<Artifact, Collection<Artifact>> expandedArtifacts = new HashMap<>();
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets = new HashMap<>();
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets = new HashMap<>();

    int missingCount = 0;
    ActionExecutionException firstActionExecutionException = null;
    MissingInputFileException missingInputException = null;
    NestedSetBuilder<Cause> rootCausesBuilder = NestedSetBuilder.stableOrder();
    ImmutableSet.Builder<Artifact> builtArtifactsBuilder = ImmutableSet.builder();
    for (Artifact input : allArtifacts) {
      try {
        SkyValue artifactValue = inputDeps.get(Artifact.key(input)).get();
        if (artifactValue != null) {
          if (artifactValue instanceof MissingFileArtifactValue) {
            missingCount++;
            final Label inputOwner = input.getOwner();
            if (inputOwner != null) {
              MissingInputFileException e =
                  ((MissingFileArtifactValue) artifactValue).getException();
              env.getListener().handle(Event.error(e.getLocation(), e.getMessage()));
              Cause cause = new LabelCause(inputOwner, e.getMessage());
              rootCausesBuilder.add(cause);
              env.getListener().handle(completor.getRootCauseError(value, cause, env));
            }
          } else {
            builtArtifactsBuilder.add(input);
            ActionInputMapHelper.addToMap(
                inputMap,
                expandedArtifacts,
                expandedFilesets,
                topLevelFilesets,
                input,
                artifactValue,
                env);
          }
        }
      } catch (ActionExecutionException e) {
        rootCausesBuilder.addTransitive(e.getRootCauses());
        // Prefer a catastrophic exception as the one we propagate.
        if (firstActionExecutionException == null
            || !firstActionExecutionException.isCatastrophe() && e.isCatastrophe()) {
          firstActionExecutionException = e;
        }
      }
    }
    expandedFilesets.putAll(topLevelFilesets);

    if (missingCount > 0) {
      missingInputException = completor.getMissingFilesException(value, missingCount, env);
      if (missingInputException == null) {
        return null;
      }
    }

    NestedSet<Cause> rootCauses = rootCausesBuilder.build();
    if (!rootCauses.isEmpty()) {
      NestedSet<ArtifactsInOutputGroup> builtOutputs =
          filterArtifactOutputGroupsToBuiltArtifacts(
              builtArtifactsBuilder.build(), artifactsToBuild);

      ExtendedEventHandler.Postable postable =
          completor.createFailed(
              value, rootCauses, builtOutputs, env, key.topLevelArtifactContext());
      if (postable == null) {
        return null;
      }
      env.getListener().post(postable);
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

    final CompletionContext ctx;
    try {
      ctx =
          CompletionContext.create(
              expandedArtifacts,
              expandedFilesets,
              key.topLevelArtifactContext().expandFilesets(),
              inputMap,
              pathResolverFactory,
              execRootSupplier.get(),
              workspaceNameValue.getName());
    } catch (IOException e) {
      throw new CompletionFunctionException(e);
    }

    ExtendedEventHandler.Postable postable =
        completor.createSucceeded(key, value, ctx, artifactsToBuild, env);
    if (postable == null) {
      return null;
    }
    env.getListener().post(postable);
    return completor.getResult();
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
