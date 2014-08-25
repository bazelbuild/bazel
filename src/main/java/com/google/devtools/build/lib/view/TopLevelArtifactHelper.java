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

package com.google.devtools.build.lib.view;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A small static class containing utility methods for handling the inclusion of
 * extra top-level artifacts into the build.
 */
public final class TopLevelArtifactHelper {

  private TopLevelArtifactHelper() {
    // Prevent instantiation.
  }

  /** Returns command-specific artifacts which may exist for a given target and build command. */
  public static final Iterable<Artifact> getCommandArtifacts(TransitiveInfoCollection target,
      String buildCommand) {
    TopLevelArtifactProvider provider = target.getProvider(TopLevelArtifactProvider.class);
    if (provider != null
        && provider.getCommandsForExtraArtifacts().contains(buildCommand.toLowerCase())) {
      return provider.getArtifactsForCommand();
    } else {
      return ImmutableList.of();
    }
  }

  /**
   * Returns all artifacts to build if this target is requested as a top-level target. The resulting
   * set includes the temps and either the files to compile, if
   * {@code options.compileOnly() == true}, or the files to run.
   *
   * <p>Calls to this method should generally return quickly; however, the runfiles computation can
   * be lazy, in which case it can be expensive on the first call. Subsequent calls may or may not
   * return the same {@code Iterable} instance.
   */
  public static NestedSet<Artifact> getAllArtifactsToBuild(TransitiveInfoCollection target,
      TopLevelArtifactContext options) {
    NestedSetBuilder<Artifact> allArtifacts = NestedSetBuilder.stableOrder();
    TempsProvider tempsProvider = target.getProvider(TempsProvider.class);
    if (tempsProvider != null) {
      allArtifacts.addAll(tempsProvider.getTemps());
    }

    TopLevelArtifactProvider topLevelArtifactProvider =
        target.getProvider(TopLevelArtifactProvider.class);
    if (topLevelArtifactProvider != null) {
      for (String outputGroup : options.outputGroups()) {
        NestedSet<Artifact> results = topLevelArtifactProvider.getOutputGroup(outputGroup);
        if (results != null) {
          allArtifacts.addTransitive(results);            
        }         
      }
    }

    if (options.compileOnly()) {
      FilesToCompileProvider provider = target.getProvider(FilesToCompileProvider.class);
      if (provider != null) {
        allArtifacts.addAll(provider.getFilesToCompile());
      }
    } else if (options.compilationPrerequisitesOnly()) {
      CompilationPrerequisitesProvider provider =
          target.getProvider(CompilationPrerequisitesProvider.class);
      if (provider != null) {
        allArtifacts.addTransitive(provider.getCompilationPrerequisites());
      }
    } else {
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

    allArtifacts.addAll(getCommandArtifacts(target, options.buildCommand()));
    return allArtifacts.build();
  }

  /**
   * Return the relative path for the target completion middleman.
   */
  public static PathFragment getMiddlemanRelativePath(Label label) {
    return new PathFragment("_middlemen").getRelative(
        Actions.escapedPath("target_complete_" + label));
  }
}
