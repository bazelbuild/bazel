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

package com.google.devtools.build.lib.actions;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Functions;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/** Helper utility to create ActionInput instances. */
public final class ActionInputHelper {
  private ActionInputHelper() {}

  @VisibleForTesting
  public static ArtifactExpander actionGraphArtifactExpander(ActionGraph actionGraph) {
    return new ArtifactExpander() {
      @Override
      public void expand(Artifact mm, Collection<? super Artifact> output) {
        // Skyframe is stricter in that it checks that "mm" is a input of the action, because
        // it cannot expand arbitrary middlemen without access to a global action graph.
        // We could check this constraint here too, but it seems unnecessary. This code is
        // going away anyway.
        Preconditions.checkArgument(mm.isMiddlemanArtifact(), "%s is not a middleman artifact", mm);
        ActionAnalysisMetadata middlemanAction = actionGraph.getGeneratingAction(mm);
        Preconditions.checkState(middlemanAction != null, mm);
        // TODO(bazel-team): Consider expanding recursively or throwing an exception here.
        // Most likely, this code will cause silent errors if we ever have a middleman that
        // contains a middleman.
        if (middlemanAction.getActionType() == MiddlemanType.AGGREGATING_MIDDLEMAN) {
          Artifact.addNonMiddlemanArtifacts(
              middlemanAction.getInputs().toList(), output, Functions.<Artifact>identity());
        }
      }
    };
  }

  /**
   * Most ActionInputs are created and never used again. On the off chance that one is, however, we
   * implement equality via path comparison. Since file caches are keyed by ActionInput, equality
   * checking does come up.
   */
  private abstract static class BasicActionInput implements ActionInput {

    // TODO(lberki): Plumb this flag from InputTree.build() somehow.
    @Override
    public boolean isSymlink() {
      return false;
    }

    @Override
    public int hashCode() {
      return getExecPathString().hashCode();
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      if (!(other instanceof BasicActionInput)) {
        return false;
      }
      return getExecPathString().equals(((BasicActionInput) other).getExecPathString());
    }

    @Override
    public String toString() {
      return "BasicActionInput: " + getExecPathString();
    }
  }

  /**
   * Creates an ActionInput with just the given relative path and no digest.
   *
   * @param path the relative path of the input.
   * @return a ActionInput.
   */
  public static ActionInput fromPath(String path) {
    return new BasicActionInput() {
      @Override
      public String getExecPathString() {
        return path;
      }

      @Override
      public PathFragment getExecPath() {
        return PathFragment.create(path);
      }
    };
  }

  /**
   * Creates an ActionInput with just the given relative path and no digest.
   *
   * @param path the relative path of the input.
   * @return a ActionInput.
   */
  public static ActionInput fromPath(PathFragment path) {
    return new BasicActionInput() {
      @Override
      public String getExecPathString() {
        return path.getPathString();
      }

      @Override
      public PathFragment getExecPath() {
        return path;
      }
    };
  }

  /**
   * Expands middleman artifacts in a sequence of {@link ActionInput}s.
   *
   * <p>Non-middleman artifacts are returned untouched.
   */
  public static List<ActionInput> expandArtifacts(
      NestedSet<? extends ActionInput> inputs, ArtifactExpander artifactExpander) {
    List<ActionInput> result = new ArrayList<>();
    List<Artifact> containedArtifacts = new ArrayList<>();
    for (ActionInput input : inputs.toList()) {
      if (!(input instanceof Artifact)) {
        result.add(input);
        continue;
      }
      containedArtifacts.add((Artifact) input);
    }
    Artifact.addExpandedArtifacts(containedArtifacts, result, artifactExpander);
    return result;
  }

  public static Iterable<String> toExecPaths(Iterable<? extends ActionInput> artifacts) {
    return Iterables.transform(artifacts, ActionInput::getExecPathString);
  }

  /** Returns the {@link Path} for an {@link ActionInput}. */
  public static Path toInputPath(ActionInput input, Path execRoot) {
    Preconditions.checkNotNull(input, "input");
    Preconditions.checkNotNull(execRoot, "execRoot");

    return (input instanceof Artifact)
        ? ((Artifact) input).getPath()
        : execRoot.getRelative(input.getExecPath());
  }
}
