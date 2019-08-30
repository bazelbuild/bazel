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
import com.google.devtools.build.lib.syntax.Type.LabelClass;
import com.google.devtools.build.lib.util.BinaryPredicate;

/**
 * A predicate that returns true if an dependency attribute should be included in the result of
 * <code>blaze query</code>.
 * Used to implement  <code>--[no]implicit_deps</code>, <code>--[no]host_deps</code> etc.
 */
public abstract class DependencyFilter
    implements BinaryPredicate<AttributeInfoProvider, Attribute> {

  /** Dependency predicate that includes all dependencies */
  public static final DependencyFilter ALL_DEPS =
      new DependencyFilter() {
        @Override
        public boolean apply(AttributeInfoProvider x, Attribute y) {
          return true;
        }
      };
  /** Dependency predicate that excludes host dependencies */
  public static final DependencyFilter NO_HOST_DEPS =
      new DependencyFilter() {
        @Override
        public boolean apply(AttributeInfoProvider infoProvider, Attribute attribute) {
          // getConfigurationTransition() is only defined for labels which introduce a dependency.
          if (attribute.getType().getLabelClass() != LabelClass.DEPENDENCY) {
            return true;
          }

          return !attribute.getTransitionFactory().isHost();
        }
      };
  /** Dependency predicate that excludes implicit dependencies */
  public static final DependencyFilter NO_IMPLICIT_DEPS =
      new DependencyFilter() {
    @Override
    public boolean apply(AttributeInfoProvider infoProvider, Attribute attribute) {
      return infoProvider.isAttributeValueExplicitlySpecified(attribute);
    }
  };
  /**
   * Dependency predicate that excludes those edges that are not present in
   * the loading phase target dependency graph.
   */
  public static final DependencyFilter NO_NODEP_ATTRIBUTES =
      new DependencyFilter() {
    @Override
    public boolean apply(AttributeInfoProvider infoProvider, Attribute attribute) {
      return attribute.getType().getLabelClass() != LabelClass.NONDEP_REFERENCE;
    }
  };
  /**
   * Checks to see if the attribute has the isDirectCompileTimeInput property.
   */
  public static final DependencyFilter DIRECT_COMPILE_TIME_INPUT =
      new DependencyFilter() {
    @Override
    public boolean apply(AttributeInfoProvider infoProvider, Attribute attribute) {
      return attribute.isDirectCompileTimeInput();
    }
  };

  /**
   * Returns true if a given attribute should be processed.
   */
  @Override
  public abstract boolean apply(AttributeInfoProvider infoProvider, Attribute attribute);

  /**
   * Returns a predicate that computes the logical and of the two given predicates.
   */
  public static DependencyFilter and(
      final DependencyFilter a, final DependencyFilter b) {
    return new DependencyFilter() {
      @Override
      public boolean apply(AttributeInfoProvider infoProvider, Attribute attribute) {
        return a.apply(infoProvider, attribute) && b.apply(infoProvider, attribute);
      }
    };
  }

  /**
   * Interface to provide information about attributes to dependency filters.
   */
  public interface AttributeInfoProvider {
    /**
     * Returns true iff the value of the specified attribute is explicitly set in
     * the BUILD file (as opposed to its default value). This also returns true if
     * the value from the BUILD file is the same as the default value.
     */
    boolean isAttributeValueExplicitlySpecified(Attribute attribute);
  }
}
