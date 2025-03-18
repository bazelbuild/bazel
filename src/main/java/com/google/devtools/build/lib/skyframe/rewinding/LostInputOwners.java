// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.MoreObjects;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;

/**
 * Association between lost inputs and the aggregation artifacts (tree artifacts, filesets,
 * runfiles) that are responsible for a failed action's inclusion of those inputs.
 */
public final class LostInputOwners {

  private final SetMultimap<ActionInput, Artifact> owners = HashMultimap.create();

  public void addOwner(ActionInput input, Artifact owner) {
    checkNotNull(input);
    checkNotNull(owner);
    checkArgument(
        isValidOwnerRelationship(input, owner),
        "Invalid owner relationship (input=%s, owner=%s)",
        input,
        owner);
    owners.put(input, owner);
  }

  private static boolean isValidOwnerRelationship(ActionInput input, Artifact owner) {
    if (owner.isFileset()) {
      return !(input instanceof Artifact);
    }
    if (owner.isTreeArtifact()) {
      return input instanceof Artifact child && child.getParent().equals(owner);
    }
    if (owner.isRunfilesTree()) {
      return input instanceof Artifact;
    }
    return false;
  }

  public ImmutableSet<Artifact> getOwners(ActionInput input) {
    return ImmutableSet.copyOf(owners.get(input));
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("owners", owners).toString();
  }
}
