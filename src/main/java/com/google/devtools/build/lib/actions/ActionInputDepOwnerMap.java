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

import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import java.util.Collection;
import javax.annotation.Nullable;

/** A {@link ActionInputDepOwners} which, as a {@link ActionInputMapSink}, is mutable. */
public class ActionInputDepOwnerMap implements ActionInputMapSink, ActionInputDepOwners {

  private final ImmutableSet<ActionInput> inputsOfInterest;
  private final Multimap<ActionInput, Artifact> depOwnersByInputs;

  public ActionInputDepOwnerMap(Collection<ActionInput> inputsOfInterest) {
    this.inputsOfInterest = ImmutableSet.copyOf(inputsOfInterest);
    this.depOwnersByInputs =
        HashMultimap.create(/*expectedKeys=*/ inputsOfInterest.size(), /*expectedValuesPerKey=*/ 1);
  }

  @Override
  public boolean put(ActionInput input, FileArtifactValue metadata, @Nullable Artifact depOwner) {
    return addOwner(input, depOwner);
  }

  public boolean addOwner(ActionInput input, @Nullable Artifact depOwner) {
    if (depOwner == null || !inputsOfInterest.contains(input)) {
      return false;
    }
    return depOwnersByInputs.put(input, depOwner);
  }

  @Override
  public ImmutableSet<Artifact> getDepOwners(ActionInput input) {
    return ImmutableSet.copyOf(depOwnersByInputs.get(input));
  }
}
