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

package com.google.devtools.build.lib.packages;

import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Pair;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Utility functions over Targets that don't really belong in the base {@link
 * Target} interface.
 */
public final class TargetUtils {

  // *_test / test_suite attribute that used to specify constraint keywords.
  private static final String CONSTRAINTS_ATTR = "tags";

  private TargetUtils() {} // Uninstantiable.

  public static boolean isTestRuleName(String name) {
    return name.endsWith("_test");
  }

  public static boolean isTestSuiteRuleName(String name) {
    return name.equals("test_suite");
  }

  /**
   * Returns true iff {@code target} is a {@code *_test} rule; excludes {@code
   * test_suite}.
   */
  public static boolean isTestRule(Target target) {
    return (target instanceof Rule) && isTestRuleName(((Rule) target).getRuleClass());
  }

  /**
   * Returns true iff {@code target} is a {@code test_suite} rule.
   */
  public static boolean isTestSuiteRule(Target target) {
    return target instanceof Rule &&
        isTestSuiteRuleName(((Rule) target).getRuleClass());
  }

  /**
   * Returns true iff {@code target} is a {@code *_test} or {@code test_suite}.
   */
  public static boolean isTestOrTestSuiteRule(Target target) {
    return isTestRule (target) || isTestSuiteRule(target);
  }

  /**
   * Returns true if {@code target} has "manual" in the tags attribute and thus should be ignored by
   * command-line wildcards or by test_suite $implicit_tests attribute.
   */
  public static boolean hasManualTag(Target target) {
    return (target instanceof Rule) && hasConstraint((Rule) target, "manual");
  }

  /**
   * Returns true if test marked as "exclusive" by the appropriate keyword
   * in the tags attribute.
   *
   * Method assumes that passed target is a test rule, so usually it should be
   * used only after isTestRule() or isTestOrTestSuiteRule(). Behavior is
   * undefined otherwise.
   */
  public static boolean isExclusiveTestRule(Rule rule) {
    return hasConstraint(rule, "exclusive");
  }

  /**
   * Returns true if test marked as "local" by the appropriate keyword
   * in the tags attribute.
   *
   * Method assumes that passed target is a test rule, so usually it should be
   * used only after isTestRule() or isTestOrTestSuiteRule(). Behavior is
   * undefined otherwise.
   */
  public static boolean isLocalTestRule(Rule rule) {
    return hasConstraint(rule, "local")
        || NonconfigurableAttributeMapper.of(rule).get("local", Type.BOOLEAN);
  }

  /**
   * Returns true if the rule is a test or test suite and is local or exclusive.
   * Wraps the above calls into one generic check safely applicable to any rule.
   */
  public static boolean isTestRuleAndRunsLocally(Rule rule) {
    return isTestOrTestSuiteRule(rule) &&
        (isLocalTestRule(rule) || isExclusiveTestRule(rule));
  }

  /**
   * Returns true if test marked as "external" by the appropriate keyword
   * in the tags attribute.
   *
   * Method assumes that passed target is a test rule, so usually it should be
   * used only after isTestRule() or isTestOrTestSuiteRule(). Behavior is
   * undefined otherwise.
   */
  public static boolean isExternalTestRule(Rule rule) {
    return hasConstraint(rule, "external");
  }

  public static List<String> getStringListAttr(Target target, String attrName) {
    Preconditions.checkArgument(target instanceof Rule);
    return NonconfigurableAttributeMapper.of((Rule) target).get(attrName, Type.STRING_LIST);
  }

  public static String getStringAttr(Target target, String attrName) {
    Preconditions.checkArgument(target instanceof Rule);
    return NonconfigurableAttributeMapper.of((Rule) target).get(attrName, Type.STRING);
  }

  public static Iterable<String> getAttrAsString(Target target, String attrName) {
    Preconditions.checkArgument(target instanceof Rule);
    List<String> values = new ArrayList<>(); // May hold null values.
    Attribute attribute = ((Rule) target).getAttributeDefinition(attrName);
    if (attribute != null) {
      Type<?> attributeType = attribute.getType();
      for (Object attrValue :
          AggregatingAttributeMapper.of((Rule) target)
              .visitAttribute(attribute.getName(), attributeType)) {

        // Ugly hack to maintain backward 'attr' query compatibility for BOOLEAN and TRISTATE
        // attributes. These are internally stored as actual Boolean or TriState objects but were
        // historically queried as integers. To maintain compatibility, we inspect their actual
        // value and return the integer equivalent represented as a String. This code is the
        // opposite of the code in BooleanType and TriStateType respectively.
        if (attributeType == BOOLEAN) {
          values.add(Type.BOOLEAN.cast(attrValue) ? "1" : "0");
        } else if (attributeType == TRISTATE) {
          switch (BuildType.TRISTATE.cast(attrValue)) {
            case AUTO:
              values.add("-1");
              break;
            case NO:
              values.add("0");
              break;
            case YES:
              values.add("1");
              break;
            default:
              throw new AssertionError("This can't happen!");
          }
        } else {
          values.add(attrValue == null ? null : attrValue.toString());
        }
      }
    }
    return values;
  }

  /**
   * If the given target is a rule, returns its <code>deprecation<code/> value, or null if unset.
   */
  @Nullable
  public static String getDeprecation(Target target) {
    if (!(target instanceof Rule)) {
      return null;
    }
    Rule rule = (Rule) target;
    return (rule.isAttrDefined("deprecation", Type.STRING))
        ? NonconfigurableAttributeMapper.of(rule).get("deprecation", Type.STRING)
        : null;
  }

  /**
   * Checks whether specified constraint keyword is present in the
   * tags attribute of the test or test suite rule.
   *
   * Method assumes that provided rule is a test or a test suite. Behavior is
   * undefined otherwise.
   */
  private static boolean hasConstraint(Rule rule, String keyword) {
    return NonconfigurableAttributeMapper.of(rule).get(CONSTRAINTS_ATTR, Type.STRING_LIST)
        .contains(keyword);
  }

  /**
   * Returns the execution info. These include execution requirement tags ('block-*', 'requires-*',
   * 'no-*', 'supports-*', 'disable-*', 'local', and 'cpu:*') as keys with empty values.
   */
  public static Map<String, String> getExecutionInfo(Rule rule) {
    // tags may contain duplicate values.
    Map<String, String> map = new HashMap<>();
    for (String tag :
        NonconfigurableAttributeMapper.of(rule).get(CONSTRAINTS_ATTR, Type.STRING_LIST)) {
      // We don't want to pollute the execution info with random things, and we also need to reserve
      // some internal tags that we don't allow to be set on targets. We also don't want to
      // exhaustively enumerate all the legal values here. Right now, only a ~small set of tags is
      // recognized by Bazel.
      if (tag.startsWith("block-")
          || tag.startsWith("requires-")
          || tag.startsWith("no-")
          || tag.startsWith("supports-")
          || tag.startsWith("disable-")
          || tag.equals("local")
          || tag.startsWith("cpu:")) {
        map.put(tag, "");
      }
    }
    return ImmutableMap.copyOf(map);
  }

  /**
   * Returns the language part of the rule name (e.g. "foo" for foo_test or foo_binary).
   *
   * <p>In practice this is the part before the "_", if any, otherwise the entire rule class name.
   *
   * <p>Precondition: isTestRule(target) || isRunnableNonTestRule(target).
   */
  public static String getRuleLanguage(Target target) {
    return getRuleLanguage(((Rule) target).getRuleClass());
  }

  /**
   * Returns the language part of the rule name (e.g. "foo" for foo_test or foo_binary).
   *
   * <p>In practice this is the part before the "_", if any, otherwise the entire rule class name.
   */
  public static String getRuleLanguage(String ruleClass) {
    int index = ruleClass.lastIndexOf('_');
    // Chop off "_binary" or "_test".
    return index != -1 ? ruleClass.substring(0, index) : ruleClass;
  }

  private static boolean isExplicitDependency(Rule rule, Label label) throws InterruptedException {
    if (rule.getVisibility().getDependencyLabels().contains(label)) {
      return true;
    }

    ExplicitEdgeVisitor visitor = new ExplicitEdgeVisitor(rule, label);
    AggregatingAttributeMapper.of(rule).visitLabels(visitor);
    return visitor.isExplicit();
  }

  /**
   * Returns a predicate to be used for test tag filtering, i.e., that only accepts tests that match
   * all of the required tags and none of the excluded tags.
   */
  public static Predicate<Target> tagFilter(List<String> tagFilterList) {
    Pair<Collection<String>, Collection<String>> tagLists =
        TestTargetUtils.sortTagsBySense(tagFilterList);
    final Collection<String> requiredTags = tagLists.first;
    final Collection<String> excludedTags = tagLists.second;
    return input -> {
      if (requiredTags.isEmpty() && excludedTags.isEmpty()) {
        return true;
      }

      if (!(input instanceof Rule)) {
        return false;
      }
      // Note that test_tags are those originating from the XX_test rule,
      // whereas the requiredTags and excludedTags originate from the command
      // line or test_suite rule.
      return TestTargetUtils.testMatchesFilters(
          ((Rule) input).getRuleTags(), requiredTags, excludedTags, false);
    };
  }

  private static class ExplicitEdgeVisitor implements AttributeMap.AcceptsLabelAttribute {
    private final Label expectedLabel;
    private final Rule rule;
    private boolean isExplicit = false;

    public ExplicitEdgeVisitor(Rule rule, Label expected) {
      this.rule = rule;
      this.expectedLabel = expected;
    }

    @Override
    public void acceptLabelAttribute(Label label, Attribute attr) {
      if (isExplicit || !rule.isAttributeValueExplicitlySpecified(attr)) {
        // Nothing to do here.
      } else if (expectedLabel.equals(label)) {
        isExplicit = true;
      }
    }

    public boolean isExplicit() {
      return isExplicit;
    }
  }

  /**
   * Return {@link Location} for {@link Target} target, if it should not be null.
   */
  public static Location getLocationMaybe(Target target) {
    return (target instanceof Rule) || (target instanceof InputFile) ? target.getLocation() : null;
  }

  /**
   * Return nicely formatted error message that {@link Label} label that was pointed to by {@link
   * Target} target did not exist, due to {@link NoSuchThingException} e.
   */
  public static String formatMissingEdge(
      @Nullable Target target, Label label, NoSuchThingException e) throws InterruptedException {
    // instanceof returns false if target is null (which is exploited here)
    if (target instanceof Rule) {
      Rule rule = (Rule) target;
      if (isExplicitDependency(rule, label)) {
        return String.format("%s and referenced by '%s'", e.getMessage(), target.getLabel());
      } else {
        // N.B. If you see this error message in one of our integration tests during development of
        // a change that adds a new implicit dependency when running Blaze, maybe you forgot to add
        // a new mock target to the integration test's setup.
        return String.format("every rule of type %s implicitly depends upon the target '%s', but "
            + "this target could not be found because of: %s", rule.getRuleClass(), label,
            e.getMessage());
      }
    } else if (target instanceof InputFile) {
      return e.getMessage() + " (this is usually caused by a missing package group in the"
          + " package-level visibility declaration)";
    } else {
      if (target != null) {
        return String.format("in target '%s', no such label '%s': %s", target.getLabel(), label,
            e.getMessage());
      }
      return e.getMessage();
    }
  }
}
