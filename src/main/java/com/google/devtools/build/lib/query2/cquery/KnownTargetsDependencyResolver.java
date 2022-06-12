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
package com.google.devtools.build.lib.query2.cquery;

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.DependencyResolver;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * KnownTargetsDependencyResolver is a DependencyResolver which resolves statically over a known set
 * of targets. It can be useful when performing queries over a known pre-resolved universe of
 * targets. This class has been extracted from TransitionsOutputFormatterCallback.java so that it
 * can be used in both ProtoOutputFormatterCallback and TransitionsOutputFormatterCallback
 */
public class KnownTargetsDependencyResolver extends DependencyResolver {

  private final ImmutableMap<Label, Target> knownTargets;

  public KnownTargetsDependencyResolver(Map<Label, Target> knownTargets) {
    this.knownTargets = ImmutableMap.copyOf(knownTargets);
  }

  @Override
  protected Map<Label, Target> getTargets(
      OrderedSetMultimap<DependencyKind, Label> labelMap,
      TargetAndConfiguration fromNode,
      NestedSetBuilder<Cause> rootCauses) {
    return labelMap.values().stream()
        .distinct()
        .filter(Objects::nonNull)
        .filter(knownTargets::containsKey)
        .collect(toImmutableMap(Function.identity(), knownTargets::get));
  }
}
