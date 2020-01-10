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

package com.google.devtools.build.lib.analysis;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.util.RegexFilter;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * A small static class containing utility methods for handling the inclusion of
 * extra top-level artifacts into the build.
 */
public final class TopLevelArtifactHelper {
  private static Logger logger = Logger.getLogger(TopLevelArtifactHelper.class.getName());

  /** Set of {@link Artifact}s in an output group. */
  @Immutable
  public static final class ArtifactsInOutputGroup {
    private final String outputGroup;
    private final boolean important;
    private final NestedSet<Artifact> artifacts;

    private ArtifactsInOutputGroup(
        String outputGroup, boolean important, NestedSet<Artifact> artifacts) {
      this.outputGroup = checkNotNull(outputGroup);
      this.important = important;
      this.artifacts = checkNotNull(artifacts);
    }

    public String getOutputGroup() {
      return outputGroup;
    }

    public NestedSet<Artifact> getArtifacts() {
      return artifacts;
    }

    /** Returns {@code true} if the user should know about this output group. */
    public boolean areImportant() {
      return important;
    }
  }

  /**
   * Returns an ArtifactsInOutputGroup instance whose artifacts are filtered to only those in a
   * given set of known-built Artifacts; used in errorful scenarios for partial output reporting.
   */
  public static ArtifactsInOutputGroup outputGroupWithOnlyBuiltArtifacts(
      ArtifactsInOutputGroup aog, ImmutableSet<Artifact> builtArtifacts) {
    NestedSetBuilder<Artifact> resultArtifacts = NestedSetBuilder.stableOrder();
    // Iterating over all artifacts in the output group although we already iterated over the set
    // while collecting all builtArtifacts. Ideally we would have a NestedSetIntersectionView that
    // would not require duplicating some-or-all of the original NestedSet.
    aog.getArtifacts().toList().stream()
        .filter(builtArtifacts::contains)
        .forEach(resultArtifacts::add);
    return new ArtifactsInOutputGroup(
        aog.getOutputGroup(), aog.areImportant(), resultArtifacts.build());
  }

  /**
   * The set of artifacts to build.
   *
   * <p>There are two kinds: the ones that the user cares about (e.g. files to build) and the ones
   * they don't (e.g. baseline coverage artifacts). The latter type doesn't get reported on various
   * outputs, e.g. on the console output listing the output artifacts of targets on the command
   * line.
   */
  @Immutable
  public static final class ArtifactsToBuild {
    private NestedSet<ArtifactsInOutputGroup> artifacts;

    private ArtifactsToBuild(NestedSet<ArtifactsInOutputGroup> artifacts) {
      this.artifacts = checkNotNull(artifacts);
    }

    /**
     * Returns the artifacts that the user should know about.
     */
    public NestedSet<Artifact> getImportantArtifacts() {
      NestedSetBuilder<Artifact> builder = new NestedSetBuilder<>(artifacts.getOrder());
      for (ArtifactsInOutputGroup artifactsInOutputGroup : artifacts.toList()) {
        if (artifactsInOutputGroup.areImportant()) {
          builder.addTransitive(artifactsInOutputGroup.getArtifacts());
        }
      }
      return builder.build();
    }

    /**
     * Returns the actual set of artifacts that need to be built.
     */
    public NestedSet<Artifact> getAllArtifacts() {
      NestedSetBuilder<Artifact> builder = new NestedSetBuilder<>(artifacts.getOrder());
      for (ArtifactsInOutputGroup artifactsInOutputGroup : artifacts.toList()) {
        builder.addTransitive(artifactsInOutputGroup.getArtifacts());
      }
      return builder.build();
    }

    /**
     * Returns the set of all {@link Artifact}s grouped by their corresponding output group.
     *
     * <p>If an {@link Artifact} belongs to two or more output groups, it appears once in each
     * output group.
     */
    public NestedSet<ArtifactsInOutputGroup> getAllArtifactsByOutputGroup() {
      return artifacts;
    }
  }

  private TopLevelArtifactHelper() {
    // Prevent instantiation.
  }

  @VisibleForTesting
  public static ArtifactsToOwnerLabels makeTopLevelArtifactsToOwnerLabels(
      AnalysisResult analysisResult, Iterable<AspectValue> aspects) {
    try (AutoProfiler ignored = AutoProfiler.logged("assigning owner labels", logger, 10)) {

      ArtifactsToOwnerLabels.Builder artifactsToOwnerLabelsBuilder =
          analysisResult.getTopLevelArtifactsToOwnerLabels().toBuilder();
    TopLevelArtifactContext artifactContext = analysisResult.getTopLevelContext();
    for (ConfiguredTarget target : analysisResult.getTargetsToBuild()) {
        addArtifactsWithOwnerLabel(
            getAllArtifactsToBuild(target, artifactContext).getAllArtifacts(),
            null,
            target.getLabel(),
            artifactsToOwnerLabelsBuilder);
    }
    for (AspectValue aspect : aspects) {
        addArtifactsWithOwnerLabel(
            getAllArtifactsToBuild(aspect, artifactContext).getAllArtifacts(),
            null,
            aspect.getLabel(),
            artifactsToOwnerLabelsBuilder);
    }
    if (analysisResult.getTargetsToTest() != null) {
      for (ConfiguredTarget target : analysisResult.getTargetsToTest()) {
          addArtifactsWithOwnerLabel(
              TestProvider.getTestStatusArtifacts(target),
              null,
              target.getLabel(),
              artifactsToOwnerLabelsBuilder);
      }
    }
      // TODO(dslomov): Artifacts to test from aspects?
      return artifactsToOwnerLabelsBuilder.build();
    }
  }

  static void addArtifactsWithOwnerLabel(
      Iterable<? extends Artifact> artifacts,
      @Nullable RegexFilter filter,
      Label ownerLabel,
      ArtifactsToOwnerLabels.Builder artifactsToOwnerLabelsBuilder) {
    for (Artifact artifact : artifacts) {
      if (filter == null || filter.isIncluded(artifact.getOwnerLabel().toString())) {
        artifactsToOwnerLabelsBuilder.addArtifact(artifact, ownerLabel);
      }
    }
  }

  /**
   * Returns all artifacts to build if this target is requested as a top-level target. The resulting
   * set includes the temps and either the files to compile, if
   * {@code context.compileOnly() == true}, or the files to run.
   *
   * <p>Calls to this method should generally return quickly; however, the runfiles computation can
   * be lazy, in which case it can be expensive on the first call. Subsequent calls may or may not
   * return the same {@code Iterable} instance.
   */
  public static ArtifactsToBuild getAllArtifactsToBuild(TransitiveInfoCollection target,
      TopLevelArtifactContext context) {
    return getAllArtifactsToBuild(
        OutputGroupInfo.get(target), target.getProvider(FileProvider.class), context);
  }

  public static ArtifactsToBuild getAllArtifactsToBuild(
      AspectValue aspectValue, TopLevelArtifactContext context) {
    ConfiguredAspect configuredAspect = aspectValue.getConfiguredAspect();
    return getAllArtifactsToBuild(
        OutputGroupInfo.get(configuredAspect),
        configuredAspect.getProvider(FileProvider.class),
        context);
  }

  static ArtifactsToBuild getAllArtifactsToBuild(
      @Nullable OutputGroupInfo outputGroupInfo,
      @Nullable FileProvider fileProvider,
      TopLevelArtifactContext context) {
    NestedSetBuilder<ArtifactsInOutputGroup> allBuilder = NestedSetBuilder.stableOrder();

    for (String outputGroup : context.outputGroups()) {
      NestedSetBuilder<Artifact> results = NestedSetBuilder.stableOrder();

      if (outputGroup.equals(OutputGroupInfo.DEFAULT) && fileProvider != null) {
        results.addTransitive(fileProvider.getFilesToBuild());
      }

      if (outputGroupInfo != null) {
        results.addTransitive(outputGroupInfo.getOutputGroup(outputGroup));
      }

      // Ignore output groups that have no artifacts.
      if (results.isEmpty()) {
        continue;
      }

      boolean isImportantGroup =
          !outputGroup.startsWith(OutputGroupInfo.HIDDEN_OUTPUT_GROUP_PREFIX);

      ArtifactsInOutputGroup artifacts =
          new ArtifactsInOutputGroup(outputGroup, isImportantGroup, results.build());

      allBuilder.add(artifacts);
    }

    return new ArtifactsToBuild(allBuilder.build());
  }
}
