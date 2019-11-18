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
//

package com.google.devtools.build.lib.bazel.rules.ninja.graph;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.List;
import java.util.Set;

public class NinjaGraphReducer {

  private final ImmutableSortedMap<PathFragment, NinjaTarget> targets;
  private ImmutableList<NinjaTarget> reducedTargets;
  private final List<PathFragment> directlyRequestedOutputs;

  public NinjaGraphReducer(Collection<NinjaTarget> targets,
      List<PathFragment> directlyRequestedOutputs) {
    ImmutableSortedMap.Builder<PathFragment, NinjaTarget> builder =
        ImmutableSortedMap.naturalOrder();
    for (NinjaTarget target : targets) {
      for (PathFragment output : target.getAllOutputs()) {
        builder.put(output, target);
      }
    }
    this.targets = builder.build();
    this.directlyRequestedOutputs = directlyRequestedOutputs;
  }

  public void reduce() {
    Set<NinjaTarget> filtered = Sets.newHashSet();
    ArrayDeque<PathFragment> queue = new ArrayDeque<>(Math.max(100, targets.size() / 4));
    queue.addAll(directlyRequestedOutputs);
    while (!queue.isEmpty()) {
      PathFragment fragment = queue.remove();
      NinjaTarget target = targets.get(fragment);
      if (target != null && filtered.add(target)) {
        queue.addAll(target.getAllInputs());
      }
    }
    reducedTargets = ImmutableList.copyOf(filtered);
  }

  public ImmutableList<NinjaTarget> getReducedTargets() {
    return reducedTargets;
  }
}
