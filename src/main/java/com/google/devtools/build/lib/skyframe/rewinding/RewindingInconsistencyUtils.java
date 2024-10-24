// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.rewinding;

import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.ArtifactNestedSetKey;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.lib.skyframe.AspectCompletionValue.AspectCompletionKey;
import com.google.devtools.build.lib.skyframe.FilesetEntryKey;
import com.google.devtools.build.lib.skyframe.TargetCompletionValue.TargetCompletionKey;
import com.google.devtools.build.lib.skyframe.TestCompletionValue.TestCompletionKey;
import com.google.devtools.build.lib.skyframe.TopLevelActionLookupKeyWrapper;
import com.google.devtools.build.lib.skyframe.TraversalRequest;
import com.google.devtools.build.skyframe.SkyKey;

/** Rewinding-related utilities used by {@link RewindableGraphInconsistencyReceiver}. */
public final class RewindingInconsistencyUtils {

  private RewindingInconsistencyUtils() {}

  static boolean mayForceRebuildChildren(SkyKey key) {
    return key instanceof ActionLookupData
        || key instanceof ArtifactNestedSetKey
        || key instanceof TopLevelActionLookupKeyWrapper;
  }

  /** Returns whether the key specifies a node which may be rewound by a failed action. */
  public static boolean isRewindable(SkyKey key) {
    return key instanceof ActionLookupData
        || key instanceof ArtifactNestedSetKey
        || key instanceof Artifact
        || key instanceof FilesetEntryKey
        || key instanceof TraversalRequest;
  }

  /**
   * Returns whether the key specifies a node which depends on nodes which may be rewound.
   *
   * <p>Such a node may discover, while in-flight, that a dependency of theirs transitioned from
   * done to undone.
   */
  static boolean isTypeThatDependsOnRewindableNodes(SkyKey key) {
    return key instanceof ActionLookupData
        || key instanceof ArtifactNestedSetKey
        || key instanceof ActionTemplateExpansionKey
        || key instanceof Artifact
        || key instanceof TargetCompletionKey
        || key instanceof TestCompletionKey
        || key instanceof AspectCompletionKey
        || key instanceof TraversalRequest
        || key instanceof FilesetEntryKey;
  }
}
