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

import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.util.BinaryPredicate;

/**
 * A predicate that returns true if an dependency attribute should be included in the result of
 * <code>blaze query</code>.
 * Used to implement  <code>--[no]implicit_deps</code>, <code>--[no]host_deps</code> etc.
 */
public abstract class DependencyFilter implements BinaryPredicate<Rule, Attribute> {

  /** Dependency predicate that includes all dependencies */
  public static final DependencyFilter ALL_DEPS =
      new DependencyFilter() {
        @Override
        public boolean apply(Rule x, Attribute y) {
          return true;
        }
      };
  /** Dependency predicate that excludes host dependencies */
  public static final DependencyFilter NO_HOST_DEPS =
      new DependencyFilter() {
    @Override
    public boolean apply(Rule rule, Attribute attribute) {
      // isHostConfiguration() is only defined for labels and label lists.
      if (attribute.getType() != BuildType.LABEL && attribute.getType() != BuildType.LABEL_LIST) {
        return true;
      }

      return attribute.getConfigurationTransition() != ConfigurationTransition.HOST;
    }
  };
  /** Dependency predicate that excludes implicit dependencies */
  public static final DependencyFilter NO_IMPLICIT_DEPS =
      new DependencyFilter() {
    @Override
    public boolean apply(Rule rule, Attribute attribute) {
      return rule.isAttributeValueExplicitlySpecified(attribute);
    }
  };
  /**
   * Dependency predicate that excludes those edges that are not present in
   * the loading phase target dependency graph.
   */
  public static final DependencyFilter NO_NODEP_ATTRIBUTES =
      new DependencyFilter() {
    @Override
    public boolean apply(Rule rule, Attribute attribute) {
      return attribute.getType() != BuildType.NODEP_LABEL
          && attribute.getType() != BuildType.NODEP_LABEL_LIST;
    }
  };
  /**
   * Checks to see if the attribute has the isDirectCompileTimeInput property.
   */
  public static final DependencyFilter DIRECT_COMPILE_TIME_INPUT =
      new DependencyFilter() {
    @Override
    public boolean apply(Rule rule, Attribute attribute) {
      return attribute.isDirectCompileTimeInput();
    }
  };

  /**
   * Returns true if a given attribute should be processed.
   */
  @Override
  public abstract boolean apply(Rule rule, Attribute attribute);

  /**
   * Returns a predicate that computes the logical and of the two given predicates.
   */
  public static DependencyFilter and(
      final DependencyFilter a, final DependencyFilter b) {
    return new DependencyFilter() {
      @Override
      public boolean apply(Rule rule, Attribute attribute) {
        return a.apply(rule, attribute) && b.apply(rule, attribute);
      }
    };
  }
}
