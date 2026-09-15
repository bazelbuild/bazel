// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.packages.DependencyFilter.AttributeInfoProvider;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import java.util.function.BiPredicate;

/**
 * A predicate that returns true if a dependency attribute should be included in the result of
 * {@code blaze query}. Used to implement {@code --[no]implicit_deps}, {@code --[no]host_deps}, etc.
 */
public interface DependencyFilter extends BiPredicate<AttributeInfoProvider, Attribute> {

  /** Dependency predicate that includes all dependencies. */
  DependencyFilter ALL_DEPS = (infoProvider, attribute) -> true;

  /** Dependency predicate that excludes non-target dependencies. */
  DependencyFilter ONLY_TARGET_DEPS = (infoProvider, attribute) -> !attribute.isToolDependency();

  /** Dependency predicate that excludes implicit dependencies. */
  DependencyFilter NO_IMPLICIT_DEPS = AttributeInfoProvider::isAttributeValueExplicitlySpecified;

  /**
   * Dependency predicate that excludes those edges that are not present in the loading phase target
   * dependency graph.
   */
  DependencyFilter NO_NODEP_ATTRIBUTES =
      (infoProvider, attribute) ->
          attribute.getType().getLabelClass() != LabelClass.NONDEP_REFERENCE;

  /**
   * Dependency predicate that excludes those edges that are not present in the loading phase target
   * dependency graph but *does* include edges from the `visibility` attribute.
   */
  DependencyFilter NO_NODEP_ATTRIBUTES_EXCEPT_VISIBILITY =
      (infoProvider, attribute) ->
          NO_NODEP_ATTRIBUTES.test(infoProvider, attribute)
              || attribute.getName().equals("visibility");

  @Override
  default DependencyFilter and(
      BiPredicate<? super AttributeInfoProvider, ? super Attribute> other) {
    return BiPredicate.super.and(other)::test;
  }

  /** Interface to provide information about attributes to dependency filters. */
  interface AttributeInfoProvider {
    /**
     * Returns true iff the value of the specified attribute is explicitly set in
     * the BUILD file (as opposed to its default value). This also returns true if
     * the value from the BUILD file is the same as the default value.
     */
    boolean isAttributeValueExplicitlySpecified(Attribute attribute);
  }
}
