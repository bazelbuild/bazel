// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/** SkyKey for {@code NestedSet<Artifact>}. */
public final class ArtifactNestedSetKey implements SkyKey {

  // TODO(jhorvitz): Consider sharing the nestedSetToSkyKey map in ArtifactNestedSetFunction.
  public static ArtifactNestedSetKey create(NestedSet<Artifact> set) {
    return new ArtifactNestedSetKey(set, set.toNode());
  }

  private final NestedSet<Artifact> set;
  private final NestedSet.Node node;

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.ARTIFACT_NESTED_SET;
  }

  ArtifactNestedSetKey(NestedSet<Artifact> set, NestedSet.Node node) {
    this.set = set;
    this.node = node;
  }

  /** Returns the set of artifacts that this key represents. */
  public NestedSet<Artifact> getSet() {
    return set;
  }

  @Override
  public int hashCode() {
    return node.hashCode();
  }

  @Override
  public boolean equals(Object that) {
    return that instanceof ArtifactNestedSetKey
        && this.node.equals(((ArtifactNestedSetKey) that).node);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("rawChildren", set).toString();
  }
}
