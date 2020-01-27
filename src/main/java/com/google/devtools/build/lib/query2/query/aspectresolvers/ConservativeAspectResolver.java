// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.query.aspectresolvers;

import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import java.util.Set;

/**
 * An aspect resolver that overestimates the required aspect dependencies.
 *
 * <p>Does not need to load any packages other than the one containing the target being processed.
 */
public class ConservativeAspectResolver implements AspectResolver {
  @Override
  public ImmutableMultimap<Attribute, Label> computeAspectDependencies(
      Target target, DependencyFilter dependencyFilter) {
    if (!(target instanceof Rule)) {
      return ImmutableMultimap.of();
    }
    Rule rule = (Rule) target;
    if (!rule.hasAspects()) {
      return ImmutableMultimap.of();
    }

    Multimap<Attribute, Label> result = LinkedHashMultimap.create();
    for (Attribute attribute : rule.getAttributes()) {
      for (Aspect aspect : attribute.getAspects(rule)) {
        AspectDefinition.addAllAttributesOfAspect(
            rule, result, aspect, dependencyFilter);
      }
    }

    return ImmutableMultimap.copyOf(result);
  }

  @Override
  public Set<Label> computeBuildFileDependencies(Package pkg) {
    // We do a conservative estimate precisely so that we don't depend on any other BUILD files.
    return ImmutableSet.copyOf(pkg.getSkylarkFileDependencies());
  }
}
