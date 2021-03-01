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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.lib.util.RegexFilter;
import java.time.Duration;
import java.util.Collection;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A small static class containing utility methods for handling the inclusion of
 * extra top-level artifacts into the build.
 */
public final class TopLevelArtifactHelper {
  /** Set of {@link Artifact}s in an output group. */
  @Immutable
  public static final class ArtifactsInOutputGroup {
    private final boolean important;
    private final NestedSet<Artifact> artifacts;

    private ArtifactsInOutputGroup(boolean important, NestedSet<Artifact> artifacts) {
      this.important = important;
      this.artifacts = checkNotNull(artifacts);
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
   * The set of artifacts to build.
   *
   * <p>There are two kinds: the ones that the user cares about (e.g. files to build) and the ones
   * they don't (e.g. baseline coverage artifacts). The latter type doesn't get reported on various
   * outputs, e.g. on the console output listing the output artifacts of targets on the command
   * line.
   */
  @Immutable
  public static final class ArtifactsToBuild {
    private final ImmutableMap<String, ArtifactsInOutputGroup> artifacts;

    private ArtifactsToBuild(ImmutableMap<String, ArtifactsInOutputGroup> artifacts) {
      this.artifacts = checkNotNull(artifacts);
    }

    /**
     * Returns the artifacts that the user should know about.
     */
    public NestedSet<Artifact> getImportantArtifacts() {
      NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
      for (ArtifactsInOutputGroup artifactsInOutputGroup : artifacts.values()) {
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
      NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
      for (ArtifactsInOutputGroup artifactsInOutputGroup : artifacts.values()) {
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
    public ImmutableMap<String, ArtifactsInOutputGroup> getAllArtifactsByOutputGroup() {
      return artifacts;
    }
  }

  private TopLevelArtifactHelper() {
    // Prevent instantiation.
  }

  private static final Duration MIN_LOGGING = Duration.ofMillis(10);

  @VisibleForTesting
  public static ArtifactsToOwnerLabels makeTopLevelArtifactsToOwnerLabels(
      AnalysisResult analysisResult) {
    try (AutoProfiler ignored =
        GoogleAutoProfilerUtils.logged("assigning owner labels", MIN_LOGGING)) {
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
      for (Map.Entry<AspectKey, ConfiguredAspect> aspectEntry :
          analysisResult.getAspectsMap().entrySet()) {
        addArtifactsWithOwnerLabel(
            getAllArtifactsToBuild(aspectEntry.getValue(), artifactContext).getAllArtifacts(),
            null,
            aspectEntry.getKey().getLabel(),
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

  public static void addArtifactsWithOwnerLabel(
      NestedSet<? extends Artifact> artifacts,
      @Nullable RegexFilter filter,
      Label ownerLabel,
      ArtifactsToOwnerLabels.Builder artifactsToOwnerLabelsBuilder) {
    addArtifactsWithOwnerLabel(
        artifacts.toList(), filter, ownerLabel, artifactsToOwnerLabelsBuilder);
  }

  public static void addArtifactsWithOwnerLabel(
      Collection<? extends Artifact> artifacts,
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
   * set includes the temps and either the files to compile, if {@code context.compileOnly() ==
   * true}, or the files to run.
   *
   * <p>Calls to this method should generally return quickly; however, the runfiles computation can
   * be lazy, in which case it can be expensive on the first call. Subsequent calls may or may not
   * return the same {@code Iterable} instance.
   */
  public static ArtifactsToBuild getAllArtifactsToBuild(
      ProviderCollection target, TopLevelArtifactContext context) {
    return getAllArtifactsToBuild(
        OutputGroupInfo.get(target),
        target.getProvider(FileProvider.class),
        context
    );
  }

  static ArtifactsToBuild getAllArtifactsToBuild(
      @Nullable OutputGroupInfo outputGroupInfo,
      @Nullable FileProvider fileProvider,
      TopLevelArtifactContext context) {
    ImmutableMap.Builder<String, ArtifactsInOutputGroup> allOutputGroups =
        ImmutableMap.builderWithExpectedSize(context.outputGroups().size());

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
          new ArtifactsInOutputGroup(isImportantGroup, results.build());

      allOutputGroups.put(outputGroup, artifacts);
    }

    return new ArtifactsToBuild(allOutputGroups.build());
  }
}
