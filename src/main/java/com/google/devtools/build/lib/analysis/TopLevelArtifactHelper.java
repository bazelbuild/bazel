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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.test.TestProvider;

/**
 * A small static class containing utility methods for handling the inclusion of
 * extra top-level artifacts into the build.
 */
public final class TopLevelArtifactHelper {

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
  public static NestedSet<Artifact> getAllArtifactsToBuild(
      Iterable<? extends TransitiveInfoCollection> targets, TopLevelArtifactContext context) {
    NestedSetBuilder<Artifact> allArtifacts = NestedSetBuilder.stableOrder();
    for (TransitiveInfoCollection target : targets) {
      allArtifacts.addTransitive(getAllArtifactsToBuild(target, context));
    }
    return allArtifacts.build();
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
  public static NestedSet<Artifact> getAllArtifactsToBuild(TransitiveInfoCollection target,
      TopLevelArtifactContext context) {
    NestedSetBuilder<Artifact> allArtifacts = NestedSetBuilder.stableOrder();
    TempsProvider tempsProvider = target.getProvider(TempsProvider.class);
    if (tempsProvider != null) {
      allArtifacts.addAll(tempsProvider.getTemps());
    }

    TopLevelArtifactProvider topLevelArtifactProvider =
        target.getProvider(TopLevelArtifactProvider.class);
    if (topLevelArtifactProvider != null) {
      for (String outputGroup : context.outputGroups()) {
        NestedSet<Artifact> results = topLevelArtifactProvider.getOutputGroup(outputGroup);
        if (results != null) {
          allArtifacts.addTransitive(results);            
        }         
      }
    }

    if (context.compileOnly()) {
      FilesToCompileProvider provider = target.getProvider(FilesToCompileProvider.class);
      if (provider != null) {
        allArtifacts.addAll(provider.getFilesToCompile());
      }
    } else if (context.compilationPrerequisitesOnly()) {
      CompilationPrerequisitesProvider provider =
          target.getProvider(CompilationPrerequisitesProvider.class);
      if (provider != null) {
        allArtifacts.addTransitive(provider.getCompilationPrerequisites());
      }
    } else if (context.buildDefaultArtifacts()) {
      FilesToRunProvider filesToRunProvider = target.getProvider(FilesToRunProvider.class);
      boolean hasRunfilesSupport = false;
      if (filesToRunProvider != null) {
        allArtifacts.addAll(filesToRunProvider.getFilesToRun());
        hasRunfilesSupport = filesToRunProvider.getRunfilesSupport() != null;
      }

      if (!hasRunfilesSupport) {
        RunfilesProvider runfilesProvider =
            target.getProvider(RunfilesProvider.class);
        if (runfilesProvider != null) {
          allArtifacts.addTransitive(runfilesProvider.getDefaultRunfiles().getAllArtifacts());
        }
      }

      AlwaysBuiltArtifactsProvider forcedArtifacts = target.getProvider(
          AlwaysBuiltArtifactsProvider.class);
      if (forcedArtifacts != null) {
        allArtifacts.addTransitive(forcedArtifacts.getArtifactsToAlwaysBuild());
      }
    }

    allArtifacts.addAll(getCoverageArtifacts(target, context));
    return allArtifacts.build();
  }

  private static Iterable<Artifact> getCoverageArtifacts(TransitiveInfoCollection target,
                                                         TopLevelArtifactContext topLevelOptions) {
    if (!topLevelOptions.compileOnly() && !topLevelOptions.compilationPrerequisitesOnly()
        && topLevelOptions.shouldRunTests()) {
      // Add baseline code coverage artifacts if we are collecting code coverage. We do that only
      // when running tests.
      // It might be slightly faster to first check if any configuration has coverage enabled.
      if (target.getConfiguration() != null
          && target.getConfiguration().isCodeCoverageEnabled()) {
        BaselineCoverageArtifactsProvider provider =
            target.getProvider(BaselineCoverageArtifactsProvider.class);
        if (provider != null) {
          return provider.getBaselineCoverageArtifacts();
        }
      }
    }
    return ImmutableList.of();
  }
}
