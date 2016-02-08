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
import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.collect.Collections2;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Helper utility to create ActionInput instances.
 */
public final class ActionInputHelper {
  private ActionInputHelper() {
  }

  @VisibleForTesting
  public static Artifact.MiddlemanExpander actionGraphMiddlemanExpander(
      final ActionGraph actionGraph) {
    return new Artifact.MiddlemanExpander() {
      @Override
      public void expand(Artifact mm, Collection<? super Artifact> output) {
        // Skyframe is stricter in that it checks that "mm" is a input of the action, because
        // it cannot expand arbitrary middlemen without access to a global action graph.
        // We could check this constraint here too, but it seems unnecessary. This code is
        // going away anyway.
        Preconditions.checkArgument(mm.isMiddlemanArtifact(),
            "%s is not a middleman artifact", mm);
        Action middlemanAction = actionGraph.getGeneratingAction(mm);
        Preconditions.checkState(middlemanAction != null, mm);
        // TODO(bazel-team): Consider expanding recursively or throwing an exception here.
        // Most likely, this code will cause silent errors if we ever have a middleman that
        // contains a middleman.
        if (middlemanAction.getActionType() == Action.MiddlemanType.AGGREGATING_MIDDLEMAN) {
          Artifact.addNonMiddlemanArtifacts(middlemanAction.getInputs(), output,
              Functions.<Artifact>identity());
        }

      }
    };
  }

  /**
   * Most ActionInputs are created and never used again. On the off chance that one is, however, we
   * implement equality via path comparison. Since file caches are keyed by ActionInput, equality
   * checking does come up.
   */
  private static class BasicActionInput implements ActionInput {
    private final String path;
    public BasicActionInput(String path) {
      this.path = Preconditions.checkNotNull(path);
    }

    @Override
    public String getExecPathString() {
      return path;
    }

    @Override
    public int hashCode() {
      return path.hashCode();
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      if (other == null) {
        return false;
      }
      if (!this.getClass().equals(other.getClass())) {
        return false;
      }
      return this.path.equals(((BasicActionInput) other).path);
    }

    @Override
    public String toString() {
      return "BasicActionInput: " + path;
    }
  }

  /**
   * Creates an ActionInput with just the given relative path and no digest.
   *
   * @param path the relative path of the input.
   * @return a ActionInput.
   */
  public static ActionInput fromPath(String path) {
    return new BasicActionInput(path);
  }

  private static final Function<String, ActionInput> FROM_PATH =
      new Function<String, ActionInput>() {
    @Override
    public ActionInput apply(String path) {
      return fromPath(path);
    }
  };

  /**
   * Creates a sequence of {@link ActionInput}s from a sequence of string paths.
   */
  public static Collection<ActionInput> fromPaths(Collection<String> paths) {
    return Collections2.transform(paths, FROM_PATH);
  }

  /**
   * Instantiates a concrete ArtifactFile with the given parent Artifact and path
   * relative to that Artifact.
   */
  public static ArtifactFile artifactFile(Artifact parent, PathFragment relativePath) {
    Preconditions.checkState(parent.isTreeArtifact(),
        "Given parent %s must be a TreeArtifact", parent);
    return new TreeArtifactFile(parent, relativePath);
  }

  /**
   * Instantiates a concrete ArtifactFile with the given parent Artifact and path
   * relative to that Artifact.
   */
  public static ArtifactFile artifactFile(Artifact parent, String relativePath) {
    return artifactFile(parent, new PathFragment(relativePath));
  }

  /** Returns an Iterable of ArtifactFiles with the given parent and parent relative paths. */
  public static Iterable<ArtifactFile> asArtifactFiles(
      final Artifact parent, Iterable<? extends PathFragment> parentRelativePaths) {
    Preconditions.checkState(parent.isTreeArtifact(),
        "Given parent %s must be a TreeArtifact", parent);
    return Iterables.transform(parentRelativePaths,
        new Function<PathFragment, ArtifactFile>() {
          @Override
          public ArtifactFile apply(PathFragment pathFragment) {
            return artifactFile(parent, pathFragment);
          }
        });
  }

  /**
   * Expands middleman artifacts in a sequence of {@link ActionInput}s.
   *
   * <p>Non-middleman artifacts are returned untouched.
   */
  public static List<ActionInput> expandMiddlemen(Iterable<? extends ActionInput> inputs,
      Artifact.MiddlemanExpander middlemanExpander) {

    List<ActionInput> result = new ArrayList<>();
    List<Artifact> containedArtifacts = new ArrayList<>();
    for (ActionInput input : inputs) {
      if (!(input instanceof Artifact)) {
        result.add(input);
        continue;
      }
      containedArtifacts.add((Artifact) input);
    }
    Artifact.addExpandedArtifacts(containedArtifacts, result, middlemanExpander);
    return result;
  }

  /** Formatter for execPath String output. Public because {@link Artifact} uses it directly. */
  public static final Function<ActionInput, String> EXEC_PATH_STRING_FORMATTER =
      new Function<ActionInput, String>() {
        @Override
        public String apply(ActionInput input) {
          return input.getExecPathString();
        }
  };

  public static Iterable<String> toExecPaths(Iterable<? extends ActionInput> artifacts) {
    return Iterables.transform(artifacts, EXEC_PATH_STRING_FORMATTER);
  }
}
