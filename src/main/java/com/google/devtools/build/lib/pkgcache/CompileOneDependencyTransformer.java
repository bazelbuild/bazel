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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.FileTarget;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.server.FailureDetails.TargetPatterns;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * Implementation of --compile_one_dependency.
 */
public final class CompileOneDependencyTransformer {
  private final TargetProvider targetProvider;

  public CompileOneDependencyTransformer(TargetProvider targetProvider) {
    this.targetProvider = targetProvider;
  }

  /**
   * For each input file in the original result, returns a rule in the same package which has the
   * input file as a source.
   */
  public ResolvedTargets<Target> transformCompileOneDependency(
      ExtendedEventHandler eventHandler, ResolvedTargets<Target> original)
      throws TargetParsingException, InterruptedException {
    if (original.hasError()) {
      return original;
    }
    ResolvedTargets.Builder<Target> builder = ResolvedTargets.builder();
    for (Target target : original.getTargets()) {
      builder.add(transformCompileOneDependency(eventHandler, target));
    }
    return builder.build();
  }

  private Target transformCompileOneDependency(ExtendedEventHandler eventHandler, Target target)
      throws TargetParsingException, InterruptedException {
    if (!(target instanceof FileTarget)) {
      throw new TargetParsingException(
          "--compile_one_dependency target '" + target.getLabel() + "' must be a file",
          TargetPatterns.Code.TARGET_MUST_BE_A_FILE);
    }

    Rule result = null;
    Iterable<Rule> orderedRuleList = getOrderedRuleList(target.getPackage());
    for (Rule rule : orderedRuleList) {
      Set<Label> labels = getInputLabels(rule);
      if (listContainsFile(eventHandler, labels, target.getLabel(), Sets.<Label>newHashSet())) {
        if (rule.getRuleClassObject().isPreferredDependency(target.getName())) {
          result = rule;
          break;
        }
        if (result == null) {
          result = rule;
        }
      }
    }

    if (result == null) {
      throw new TargetParsingException(
          "Couldn't find dependency on target '" + target.getLabel() + "'",
          TargetPatterns.Code.DEPENDENCY_NOT_FOUND);
    }

    // TODO(djasper): Check whether parse_headers is disabled and just return if not.
    // If the rule has source targets, return it.
    if (result.getRuleClassObject().hasAttr("srcs", BuildType.LABEL_LIST)
        && !RawAttributeMapper.of(result).getMergedValues("srcs", BuildType.LABEL_LIST).isEmpty()) {
      return result;
    }

    // Try to find a rule in the same package that has 'result' as a dependency.
    for (Rule rule : orderedRuleList) {
      RawAttributeMapper attributes = RawAttributeMapper.of(rule);
      // We don't know which path to follow for configurable attributes, so skip them.
      if (attributes.isConfigurable("deps") || attributes.isConfigurable("srcs")) {
        continue;
      }
      RuleClass ruleClass = rule.getRuleClassObject();
      if (ruleClass.hasAttr("deps", BuildType.LABEL_LIST)
          && ruleClass.hasAttr("srcs", BuildType.LABEL_LIST)) {
        for (Label dep : attributes.get("deps", BuildType.LABEL_LIST)) {
          if (dep.equals(result.getLabel())) {
            if (!attributes.get("srcs", BuildType.LABEL_LIST).isEmpty()) {
              return rule;
            }
          }
        }
      }
    }

    return result;
  }

  /**
   * Returns a list of rules in the given package sorted by BUILD file order. When
   * multiple rules depend on a target, we choose the first match in this list (after
   * filtering for preferred dependencies - see below).
   */
  private Iterable<Rule> getOrderedRuleList(Package pkg) {
    List<Rule> orderedList = Lists.newArrayList();
    for (Rule rule : pkg.getTargets(Rule.class)) {
      orderedList.add(rule);
    }

    Collections.sort(orderedList, Comparator.comparing(arg -> arg.getLocation()));
    return orderedList;
  }

  /**
   * Returns true if a specific rule compiles a specific source. Looks through genrules and
   * filegroups.
   */
  private boolean listContainsFile(
      ExtendedEventHandler eventHandler,
      Collection<Label> srcLabels,
      Label source,
      Set<Label> visitedRuleLabels)
      throws TargetParsingException, InterruptedException {
    if (srcLabels.contains(source)) {
      return true;
    }
    for (Label label : srcLabels) {
      if (!visitedRuleLabels.add(label)) {
        continue;
      }
      Target target = null;
      try {
        target = targetProvider.getTarget(eventHandler, label);
      } catch (NoSuchThingException e) {
        // Just ignore failing sources/packages. We could report them here, but as long as we do
        // early return, the presence of this error would then be determined by the order of items
        // in the srcs attribute. A proper error will be created by the subsequent loading.
      }
      if (target == null || target instanceof FileTarget) {
        continue;
      }
      Rule targetRule = target.getAssociatedRule();
      if ("filegroup".equals(targetRule.getRuleClass())) {
        RawAttributeMapper attributeMapper = RawAttributeMapper.of(targetRule);
        Collection<Label> srcs = attributeMapper.getMergedValues("srcs", BuildType.LABEL_LIST);
        if (listContainsFile(eventHandler, srcs, source, visitedRuleLabels)) {
          return true;
        }
      } else if ("genrule".equals(targetRule.getRuleClass())) {
        // TODO(djasper): Likely, it makes much more sense to look at the inputs of a genrule.
        for (OutputFile file : targetRule.getOutputFiles()) {
          if (file.getLabel().equals(source)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  /** Returns all labels that are contained in direct compile time inputs of {@code rule}. */
  private static Set<Label> getInputLabels(Rule rule) {
    RawAttributeMapper attributeMapper = RawAttributeMapper.of(rule);
    Set<Label> labels = new TreeSet<>();
    for (String attrName : attributeMapper.getAttributeNames()) {
      if (!attributeMapper.getAttributeDefinition(attrName).isDirectCompileTimeInput()) {
        continue;
      }
      // TODO(djasper): We might also want to look at LABEL types, but there currently is the
      // attribute xcode_config, which leads to test errors in Bazel tests.
      if (rule.isAttrDefined(attrName, BuildType.LABEL_LIST)) {
        labels.addAll(attributeMapper.getMergedValues(attrName, BuildType.LABEL_LIST));
      }
    }
    return labels;
  }
}
