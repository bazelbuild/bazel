// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSet.Builder;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/** Helper methods that are shared by the different sandboxing strategies in this package. */
public final class SandboxHelpers {

  /**
   * Returns the inputs of a Spawn as a map of PathFragments relative to an execRoot to paths in the
   * host filesystem where the input files can be found.
   */
  public static Map<PathFragment, Path> getInputFiles(
      SpawnInputExpander spawnInputExpander,
      Path execRoot,
      Spawn spawn,
      ActionExecutionContext executionContext)
      throws IOException {
    Map<PathFragment, ActionInput> inputMap =
        spawnInputExpander.getInputMapping(
            spawn,
            executionContext.getArtifactExpander(),
            executionContext.getActionInputFileCache(),
            executionContext.getContext(FilesetActionContext.class));
    return postProcess(inputMap, spawn, executionContext.getArtifactExpander(), execRoot);
  }

  /**
   * Returns the inputs of a Spawn as a map of PathFragments relative to an execRoot to paths in the
   * host filesystem where the input files can be found.
   */
  public static Map<PathFragment, Path> getInputFiles(
      Spawn spawn,
      SpawnExecutionPolicy policy,
      Path execRoot)
          throws IOException {
    return postProcess(policy.getInputMapping(), spawn, policy.getArtifactExpander(), execRoot);
  }

  /**
   * Returns the inputs of a Spawn as a map of PathFragments relative to an execRoot to paths in the
   * host filesystem where the input files can be found.
   */
  private static Map<PathFragment, Path> postProcess(
      Map<PathFragment, ActionInput> inputMap,
      Spawn spawn, 
      ArtifactExpander artifactExpander,
      Path execRoot) {
    // SpawnInputExpander#getInputMapping uses ArtifactExpander#expandArtifacts to expand
    // middlemen and tree artifacts, which expands empty tree artifacts to no entry. However,
    // actions that accept TreeArtifacts as inputs generally expect that the empty directory is
    // created. So we add those explicitly here.
    // TODO(ulfjack): Move this code to SpawnInputExpander.
    for (ActionInput input : spawn.getInputFiles()) {
      if (input instanceof Artifact && ((Artifact) input).isTreeArtifact()) {
        List<Artifact> containedArtifacts = new ArrayList<>();
        artifactExpander.expand((Artifact) input, containedArtifacts);
        // Attempting to mount a non-empty directory results in ERR_DIRECTORY_NOT_EMPTY, so we
        // only mount empty TreeArtifacts as directories.
        if (containedArtifacts.isEmpty()) {
          inputMap.put(input.getExecPath(), input);
        }
      }
    }

    Map<PathFragment, Path> inputFiles = new TreeMap<>();
    for (Map.Entry<PathFragment, ActionInput> e : inputMap.entrySet()) {
      Path inputPath =
          e.getValue() == SpawnInputExpander.EMPTY_FILE
              ? null
              : execRoot.getRelative(e.getValue().getExecPath());
      inputFiles.put(e.getKey(), inputPath);
    }
    return inputFiles;
  }

  public static ImmutableSet<PathFragment> getOutputFiles(Spawn spawn) {
    Builder<PathFragment> outputFiles = ImmutableSet.builder();
    for (ActionInput output : spawn.getOutputFiles()) {
      outputFiles.add(PathFragment.create(output.getExecPathString()));
    }
    return outputFiles.build();
  }

  /**
   * Returns true if the build options are set in a way that requires network access for all
   * actions. This is separate from {@link Spawns#requiresNetwork} to avoid having to keep a
   * reference to the full set of build options (and also for performance, since this only needs to
   * be checked once-per-build).
   */
  static boolean shouldAllowNetwork(OptionsProvider buildOptions) {
    // Allow network access, when --java_debug is specified, otherwise we can't connect to the
    // remote debug server of the test. This intentionally overrides the "block-network" execution
    // tag.
    return buildOptions
        .getOptions(TestConfiguration.TestOptions.class)
        .testArguments
        .contains("--wrapper_script_flag=--debug");
  }
}
