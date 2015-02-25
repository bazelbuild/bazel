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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Helper for actions that do include scanning. Currently only deals with source files, so is only
 * appropriate for actions that do not discover generated files. Currently does not do .d file
 * parsing, so the set of artifacts returned may be an overapproximation to the ones actually used
 * during execution.
 */
public class DiscoveredSourceInputsHelper {

  private DiscoveredSourceInputsHelper() {
  }

  /**
   * Converts PathFragments into source Artifacts using an ArtifactResolver, ignoring any that are
   * already in mandatoryInputs. Silently drops any PathFragments that cannot be resolved into
   * Artifacts.
   */
  public static ImmutableList<Artifact> getDiscoveredInputsFromPaths(
      Iterable<Artifact> mandatoryInputs, ArtifactResolver artifactResolver,
      Collection<PathFragment> inputPaths) {
    Set<PathFragment> knownPathFragments = new HashSet<>();
    for (Artifact input : mandatoryInputs) {
      knownPathFragments.add(input.getExecPath());
    }
    ImmutableList.Builder<Artifact> foundInputs = ImmutableList.builder();
    for (PathFragment execPath : inputPaths) {
      if (!knownPathFragments.add(execPath)) {
        // Don't add any inputs that we already added, or original inputs, which we probably
        // couldn't convert into artifacts anyway.
        continue;
      }
      Artifact artifact = artifactResolver.resolveSourceArtifact(execPath);
      // It is unlikely that this artifact is null, but tolerate the situation just in case.
      // It is safe to ignore such paths because dependency checker would identify change in inputs
      // (ignored path was used before) and will force action execution.
      if (artifact != null) {
        foundInputs.add(artifact);
      }
    }
    return foundInputs.build();
  }

  /**
   * Converts ActionInputs discovered as inputs during execution into source Artifacts, ignoring any
   * that are already in mandatoryInputs or that live in builtInIncludeDirectories. If any
   * ActionInputs cannot be resolved, an ActionExecutionException will be thrown.
   *
   * <p>This method duplicates the functionality of CppCompileAction#populateActionInputs, though it
   * is simpler because it need not deal with derived artifacts and doesn't parse the .d file.
   */
  public static ImmutableList<Artifact> getDiscoveredInputsFromActionInputs(
      Iterable<Artifact> mandatoryInputs,
      ArtifactResolver artifactResolver,
      Iterable<? extends ActionInput> discoveredInputs,
      Iterable<PathFragment> builtInIncludeDirectories,
      Action action,
      Artifact primaryInput) throws ActionExecutionException {
    List<PathFragment> systemIncludePrefixes = new ArrayList<>();
    for (PathFragment includePath : builtInIncludeDirectories) {
      if (includePath.isAbsolute()) {
        systemIncludePrefixes.add(includePath);
      }
    }

    // Avoid duplicates by keeping track of the ones we've seen so far, even though duplicates are
    // unlikely, since they would have to be inputs to this (non-CppCompile) action and also
    // #included by a C++ source file.
    Set<Artifact> knownInputs = new HashSet<>();
    Iterables.addAll(knownInputs, mandatoryInputs);
    ImmutableList.Builder<Artifact> foundInputs = ImmutableList.builder();
    // Check inclusions.
    IncludeProblems problems = new IncludeProblems();
    for (ActionInput input : discoveredInputs) {
      if (input instanceof Artifact) {
        Artifact artifact = (Artifact) input;
        if (knownInputs.add(artifact)) {
          foundInputs.add(artifact);
        }
        continue;
      }
      PathFragment execPath = new PathFragment(input.getExecPathString());
      if (execPath.isAbsolute()) {
        // Absolute includes from system paths are ignored.
        if (FileSystemUtils.startsWithAny(execPath, systemIncludePrefixes)) {
          continue;
        }
        // Theoretically, the more sophisticated logic of CppCompileAction#populateActioInputs could
        // be used here, to allow absolute includes that started with the execRoot. However, since
        // we don't hit this codepath for local execution, that should be unnecessary. If and when
        // we examine the results of local execution for scanned includes, that case may need to be
        // dealt with.
        problems.add(execPath.getPathString());
      }
      Artifact artifact = artifactResolver.resolveSourceArtifact(execPath);
      if (artifact != null) {
        if (knownInputs.add(artifact)) {
          foundInputs.add(artifact);
        }
      } else {
        // Abort if we see files that we can't resolve, likely caused by
        // undeclared includes or illegal include constructs.
        problems.add(execPath.getPathString());
      }
    }
    problems.assertProblemFree(action, primaryInput);
    return foundInputs.build();
  }
}
