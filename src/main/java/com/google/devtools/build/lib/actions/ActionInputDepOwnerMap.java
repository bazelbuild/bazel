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
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.SetMultimap;

/** A mutable {@link ActionInputDepOwners}. */
public final class ActionInputDepOwnerMap implements ActionInputDepOwners {

  private final SetMultimap<ActionInput, Artifact> depOwnersByInputs = HashMultimap.create();

  public void addOwner(ActionInput input, Artifact owner) {
    checkNotNull(input);
    checkNotNull(owner);
    checkArgument(
        isValidOwnerRelationship(input, owner),
        "Invalid owner relationship (input=%s, owner=%s)",
        input,
        owner);
    depOwnersByInputs.put(input, owner);
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

  @Override
  public ImmutableSet<Artifact> getDepOwners(ActionInput input) {
    return ImmutableSet.copyOf(depOwnersByInputs.get(input));
  }
}
