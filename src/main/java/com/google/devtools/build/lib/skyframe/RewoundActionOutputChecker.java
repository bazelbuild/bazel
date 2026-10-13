// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import java.util.Arrays;
import javax.annotation.Nullable;

/** Content-only comparison against the same action's retained value from before rewinding. */
final class RewoundActionOutputChecker {
  private RewoundActionOutputChecker() {}

  /**
   * Returns the path of a changed output or tree child, or null if its content is unchanged.
   *
   * <p>Storage location and filesystem proxies may change when remote outputs are materialized.
   * Tree archives are transport representations, not additional logical outputs. Neither is part of
   * this comparison. Rich artifact data and digestless contents are not checked.
   */
  @Nullable
  static String getChangedOutput(
      Artifact output,
      ActionExecutionValue previous,
      @Nullable FileArtifactValue metadata,
      @Nullable TreeArtifactValue tree) {
    if (!output.isTreeArtifact()) {
      return sameContent(previous.getAllFileValues().get(output), metadata)
          ? null
          : output.getExecPathString();
    }
    TreeArtifactValue previousTree = previous.getAllTreeArtifactValues().get(output);
    if (previousTree == null || tree == null) {
      return previousTree == tree ? null : output.getExecPathString();
    }
    var previousChildren = previousTree.getChildValues();
    var children = tree.getChildValues();
    for (var entry : previousChildren.entrySet()) {
      if (!sameContent(entry.getValue(), children.get(entry.getKey()))) {
        return entry.getKey().getExecPathString();
      }
    }
    for (var child : children.keySet()) {
      if (!previousChildren.containsKey(child)) {
        return child.getExecPathString();
      }
    }
    return null;
  }

  private static boolean sameContent(
      @Nullable FileArtifactValue previous, @Nullable FileArtifactValue current) {
    if (previous == null || current == null) {
      return previous == current;
    }
    return previous.getType() == current.getType()
        && Arrays.equals(previous.getDigest(), current.getDigest())
        && (previous.getDigest() == null || previous.getSize() == current.getSize());
  }
}
