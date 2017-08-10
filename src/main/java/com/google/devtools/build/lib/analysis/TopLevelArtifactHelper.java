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

import static com.google.devtools.build.lib.util.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.AspectValue;
import javax.annotation.Nullable;

/**
 * A small static class containing utility methods for handling the inclusion of
 * extra top-level artifacts into the build.
 */
public final class TopLevelArtifactHelper {

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
   * The set of artifacts to build.
   *
   * <p>There are two kinds: the ones that the user cares about (e.g. files to build) and the ones
   * she doesn't (e.g. baseline coverage artifacts). The latter type doesn't get reported on various
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
      for (ArtifactsInOutputGroup artifactsInOutputGroup : artifacts) {
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
      for (ArtifactsInOutputGroup artifactsInOutputGroup : artifacts) {
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

  /**
   * Utility function to form a list of all test output Artifacts of the given targets to test.
   */
  public static ImmutableCollection<Artifact> getAllArtifactsToTest(
      Iterable<? extends TransitiveInfoCollection> targets) {
    if (targets == null) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<Artifact> allTestArtifacts = ImmutableList.builder();
    for (TransitiveInfoCollection target : targets) {
      allTestArtifacts.addAll(TestProvider.getTestStatusArtifacts(target));
    }
    return allTestArtifacts.build();
  }

  /**
   * Utility function to form a NestedSet of all top-level Artifacts of the given targets.
   */
  public static ArtifactsToBuild getAllArtifactsToBuild(
      Iterable<? extends TransitiveInfoCollection> targets, TopLevelArtifactContext context) {
    NestedSetBuilder<ArtifactsInOutputGroup> artifacts = NestedSetBuilder.stableOrder();
    for (TransitiveInfoCollection target : targets) {
      ArtifactsToBuild targetArtifacts = getAllArtifactsToBuild(target, context);
      artifacts.addTransitive(targetArtifacts.getAllArtifactsByOutputGroup());
    }
    return new ArtifactsToBuild(artifacts.build());
  }

  /**
   * Utility function to form a NestedSet of all top-level Artifacts of the given targets.
   */
  public static ArtifactsToBuild getAllArtifactsToBuildFromAspects(
      Iterable<AspectValue> aspects, TopLevelArtifactContext context) {
    NestedSetBuilder<ArtifactsInOutputGroup> artifacts = NestedSetBuilder.stableOrder();
    for (AspectValue aspect : aspects) {
      ArtifactsToBuild aspectArtifacts = getAllArtifactsToBuild(aspect, context);
      artifacts.addTransitive(aspectArtifacts.getAllArtifactsByOutputGroup());
    }
    return new ArtifactsToBuild(artifacts.build());
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
        OutputGroupProvider.get(target),
        target.getProvider(FileProvider.class),
        context
    );
  }

  public static ArtifactsToBuild getAllArtifactsToBuild(
      AspectValue aspectValue, TopLevelArtifactContext context) {
    ConfiguredAspect configuredAspect = aspectValue.getConfiguredAspect();
    return getAllArtifactsToBuild(
        OutputGroupProvider.get(configuredAspect),
        configuredAspect.getProvider(FileProvider.class),
        context);
  }

  public static ArtifactsToBuild getAllArtifactsToBuild(
      @Nullable OutputGroupProvider outputGroupProvider,
      @Nullable FileProvider fileProvider,
      TopLevelArtifactContext context) {
    NestedSetBuilder<ArtifactsInOutputGroup> allBuilder = NestedSetBuilder.stableOrder();

    for (String outputGroup : context.outputGroups()) {
      NestedSetBuilder<Artifact> results = NestedSetBuilder.stableOrder();

      if (outputGroup.equals(OutputGroupProvider.DEFAULT) && fileProvider != null) {
        results.addTransitive(fileProvider.getFilesToBuild());
      }

      if (outputGroupProvider != null) {
        results.addTransitive(outputGroupProvider.getOutputGroup(outputGroup));
      }

      // Ignore output groups that have no artifacts.
      if (results.isEmpty()) {
        continue;
      }

      boolean isImportantGroup =
          !outputGroup.startsWith(OutputGroupProvider.HIDDEN_OUTPUT_GROUP_PREFIX);

      ArtifactsInOutputGroup artifacts =
          new ArtifactsInOutputGroup(outputGroup, isImportantGroup, results.build());

      allBuilder.add(artifacts);
    }

    return new ArtifactsToBuild(allBuilder.build());
  }
}
