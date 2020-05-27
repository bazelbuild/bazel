// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.ninja.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Helper class for caching computation of transitive inclusion of non-phony targets into phony
 * ones. We cannot compute all artifacts for all phony targets because some of them may not be
 * created by a subgraph of required actions.
 */
@ThreadSafe
public class PhonyTargetArtifacts {
  private final Map<PathFragment, NestedSet<Artifact>> cache;
  private final ImmutableSortedMap<PathFragment, PhonyTarget> phonyTargetsMap;
  private final NinjaGraphArtifactsHelper artifactsHelper;

  public PhonyTargetArtifacts(
      ImmutableSortedMap<PathFragment, PhonyTarget> phonyTargetsMap,
      NinjaGraphArtifactsHelper artifactsHelper) {
    this.phonyTargetsMap = phonyTargetsMap;
    this.artifactsHelper = artifactsHelper;
    cache = Maps.newHashMap();
  }

  NestedSet<Artifact> getPhonyTargetArtifacts(PathFragment name) throws GenericParsingException {
    NestedSet<Artifact> existing = cache.get(name);
    if (existing != null) {
      return existing;
    }
    PhonyTarget phonyTarget = phonyTargetsMap.get(name);
    Preconditions.checkNotNull(phonyTarget);
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    for (PathFragment input : phonyTarget.getDirectExplicitInputs()) {
      builder.add(artifactsHelper.getInputArtifact(input));
    }
    for (PathFragment phonyName : phonyTarget.getPhonyNames()) {
      // We already checked for cycles during loading.
      NestedSet<Artifact> nestedSet = getPhonyTargetArtifacts(phonyName);
      builder.addTransitive(nestedSet);
    }
    NestedSet<Artifact> value = builder.build();
    // We do not hold the lock during the computation, so deadlocks are not possible,
    // however duplicate computations are possible.
    cache.put(name, value);
    return value;
  }
}
