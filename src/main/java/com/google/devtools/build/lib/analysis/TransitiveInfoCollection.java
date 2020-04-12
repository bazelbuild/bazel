// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RequiredProviders;
import com.google.devtools.build.lib.skylarkbuildapi.core.TransitiveInfoCollectionApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.SkylarkIndexable;

/**
 * Multiple {@link TransitiveInfoProvider}s bundled together.
 *
 * <p>Represents the information made available by a {@link ConfiguredTarget} to other ones that
 * depend on it. For more information about the analysis phase, see {@link
 * com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory}.
 *
 * <p>Implementations of build rules should <b>not</b> hold on to references to the {@link
 * TransitiveInfoCollection}s representing their direct prerequisites in order to reduce their
 * memory footprint (otherwise, the referenced object could refer one of its direct dependencies in
 * turn, thereby making the size of the objects reachable from a single instance unbounded).
 *
 * @see com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory
 * @see TransitiveInfoProvider
 */
public interface TransitiveInfoCollection
    extends SkylarkIndexable, ProviderCollection, TransitiveInfoCollectionApi {

  @Override
  default Depset outputGroup(String group) {
    OutputGroupInfo provider = OutputGroupInfo.get(this);
    NestedSet<Artifact> result = provider != null
        ? provider.getOutputGroup(group)
        : NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER);
    return Depset.of(Artifact.TYPE, result);
  }

  /**
   * Returns the label associated with this prerequisite.
   */
  Label getLabel();

  /**
   * Checks whether this {@link TransitiveInfoCollection} satisfies given {@link RequiredProviders}.
   */
  default boolean satisfies(RequiredProviders providers) {
    return providers.isSatisfiedBy(
        aClass -> getProvider(aClass.asSubclass(TransitiveInfoProvider.class)) != null,
        id -> this.get(id) != null);
  }

  /**
   * Returns providers that this {@link TransitiveInfoCollection} misses from a given {@link
   * RequiredProviders}.
   *
   * <p>If none are missing, returns {@link RequiredProviders} that accept any set of providers.
   */
  default RequiredProviders missingProviders(RequiredProviders providers) {
    return providers.getMissing(
        aClass -> getProvider(aClass.asSubclass(TransitiveInfoProvider.class)) != null,
        id -> this.get(id) != null);
  }
}
