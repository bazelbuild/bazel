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

package com.google.devtools.build.lib.generatedprojecttest.util;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.Pair;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * Utility class for providing static predicates for rules, to help filter the rules.
 */
public class RuleSetUtils {

  /**
   * Predicate for checking if a rule is hidden.
   */
  public static final Predicate<String> HIDDEN_RULE = new Predicate<String>() {
    @Override
    public boolean apply(final String input) {
      try {
        RuleClassType.INVISIBLE.checkName(input);
        return true;
      } catch (IllegalArgumentException e) {
        return input.equals("testing_dummy_rule")
            || input.equals("testing_rule_for_mandatory_providers");
      }
    }
  };

  /** Predicate for checking if a rule has any mandatory attributes, aside from name. */
  public static final Predicate<RuleClass> MANDATORY_ATTRIBUTES =
      new Predicate<RuleClass>() {
        @Override
        public boolean apply(final RuleClass input) {
          List<Attribute> li = new ArrayList<>(input.getAttributes());
          return Iterables.any(li, RuleSetUtils::mandatoryExcludingName);
        }
      };

  /**
   * Predicate for checking that the rule can have a deps attribute, and does not have any other
   * mandatory attributes besides deps and name.
   */
  public static final Predicate<RuleClass> DEPS_ONLY_ALLOWED =
      new Predicate<RuleClass>() {
        @Override
        public boolean apply(final RuleClass input) {
          List<Attribute> li = new ArrayList<>(input.getAttributes());
          // TODO(bazel-team): after the API migration we shouldn't check srcs separately
          boolean emptySrcsAllowed =
              input.hasAttr("srcs", BuildType.LABEL_LIST)
                  ? !input.getAttributeByName("srcs").isNonEmpty()
                  : true;
          if (!(emptySrcsAllowed && Iterables.any(li, DEPS))) {
            return false;
          }

          Iterator<Attribute> it = li.iterator();
          boolean mandatoryAttributesBesidesDeps =
              Iterables.any(
                  Lists.newArrayList(Iterators.filter(it, RuleSetUtils::mandatoryExcludingName)),
                  Predicates.not(DEPS));
          return !mandatoryAttributesBesidesDeps;
        }
      };

  /**
   * Predicate for checking if a RuleClass has certain attributes
   */
  public static class HasAttributes implements Predicate<RuleClass> {

    private final List<Pair<String, Type<?>>> attributes;

    public HasAttributes(Collection<Pair<String, Type<?>>> attributes) {
      this.attributes = ImmutableList.copyOf(attributes);
    }

    @Override
    public boolean apply(final RuleClass ruleClass) {
      return attributes.stream().anyMatch(pair -> ruleClass.hasAttr(pair.first, pair.second));
    }
  }

  public static Predicate<RuleClass> hasAnyAttributes(
      Collection<Pair<String, Type<?>>> attributes) {
    return new HasAttributes(attributes);
  }

  /** Predicate for checking if an attribute (other than name) is mandatory. */
  private static boolean mandatoryExcludingName(Attribute input) {
    return input.isMandatory() && !input.getName().equals("name");
  }

  /**
   * Predicate for checking if an attribute is the "deps" attribute.
   */
  private static final Predicate<Attribute> DEPS = new Predicate<Attribute>() {
    @Override
    public boolean apply(final Attribute input) {
      return input.getName().equals("deps");
    }
  };

  /**
   * Predicate for checking if a rule class is not in excluded.
   */
  public static Predicate<String> notContainsAnyOf(final ImmutableSet<String> excluded) {
    return Predicates.not(Predicates.in(excluded));
  }
}
