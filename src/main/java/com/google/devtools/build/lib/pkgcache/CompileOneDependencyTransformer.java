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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.FileTarget;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Implementation of --compile_one_dependency.
 */
final class CompileOneDependencyTransformer {

  private final PackageManager pkgManager;

  public CompileOneDependencyTransformer(PackageManager pkgManager) {
    this.pkgManager = pkgManager;
  }

  /**
   * For each input file in the original result, returns a rule in the same package which has the
   * input file as a source.
   */
  public ResolvedTargets<Target> transformCompileOneDependency(EventHandler eventHandler,
      ResolvedTargets<Target> original) throws TargetParsingException {
    if (original.hasError()) {
      return original;
    }
    ResolvedTargets.Builder<Target> builder = ResolvedTargets.builder();
    for (Target target : original.getTargets()) {
      builder.add(transformCompileOneDependency(eventHandler, target));
    }
    return builder.build();
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

    Collections.sort(orderedList, new Comparator<Rule>() {
      @Override
      public int compare(Rule o1, Rule o2) {
        return Integer.compare(
            o1.getLocation().getStartOffset(),
            o2.getLocation().getStartOffset());
      }
    });
    return orderedList;
  }

  private Target transformCompileOneDependency(EventHandler eventHandler, Target target)
      throws TargetParsingException {
    if (!(target instanceof FileTarget)) {
      throw new TargetParsingException("--compile_one_dependency target '" +
                                       target.getLabel() + "' must be a file");
    }

    Package pkg = target.getPackage();

    Iterable<Rule> orderedRuleList = getOrderedRuleList(pkg);
    // Consuming rule to return if no "preferred" rules have been found.
    Rule fallbackRule = null;

    for (Rule rule : orderedRuleList) {
      try {
        // The call to getSrcTargets here can be removed in favor of the
        // rule.getLabels() call below once we update "srcs" for all rules.
        if (SrcTargetUtil.getSrcTargets(eventHandler, rule, pkgManager).contains(target)) {
          if (rule.getRuleClassObject().isPreferredDependency(target.getName())) {
            return rule;
          } else if (fallbackRule == null) {
            fallbackRule = rule;
          }
        }
      } catch (NoSuchThingException e) {
        // Nothing to see here. Move along.
      } catch (InterruptedException e) {
        throw new TargetParsingException("interrupted");
      }
    }

    Rule result = null;

    // For each rule, see if it has directCompileTimeInputAttribute,
    // and if so check the targets listed in that attribute match the label.
    DependencyFilter directCompileTimeInput =
        new DependencyFilter() {
          @Override
          public boolean apply(Rule rule, Attribute attribute) {
            return DependencyFilter.DIRECT_COMPILE_TIME_INPUT.apply(rule, attribute)
                // We don't know which path to follow for configurable attributes, so skip them.
                && !rule.isConfigurableAttribute(attribute.getName());
          }
        };
    for (Rule rule : orderedRuleList) {
      if (rule.getLabels(directCompileTimeInput).contains(target.getLabel())) {
        if (rule.getRuleClassObject().isPreferredDependency(target.getName())) {
          result = rule;
        } else if (fallbackRule == null) {
          fallbackRule = rule;
        }
      }
    }

    if (result == null) {
      result = fallbackRule;
    }

    if (result == null) {
      throw new TargetParsingException(
          "Couldn't find dependency on target '" + target.getLabel() + "'");
    }

    try {
      // If the rule has source targets, return it.
      if (!SrcTargetUtil.getSrcTargets(eventHandler, result, pkgManager).isEmpty()) {
        return result;
      }
    } catch (NoSuchThingException e) {
      throw new TargetParsingException(
          "Couldn't find dependency on target '" + target.getLabel() + "'");
    } catch (InterruptedException e) {
      throw new TargetParsingException("interrupted");
    }

    for (Rule rule : orderedRuleList) {
      RawAttributeMapper attributes = RawAttributeMapper.of(rule);
      // We don't know which path to follow for configurable attributes, so skip them.
      if (attributes.isConfigurable("deps", BuildType.LABEL_LIST)
          || attributes.isConfigurable("srcs", BuildType.LABEL_LIST)) {
        continue;
      }
      RuleClass ruleClass = rule.getRuleClassObject();
      if (ruleClass.hasAttr("deps", BuildType.LABEL_LIST) &&
          ruleClass.hasAttr("srcs", BuildType.LABEL_LIST)) {
        for (Label dep : attributes.get("deps", BuildType.LABEL_LIST)) {
          if (dep.equals(result.getLabel())) {
            if (!attributes.get("srcs", BuildType.LABEL_LIST).isEmpty()) {
              return rule;
            }
          }
        }
      }
    }

    throw new TargetParsingException(
        "Couldn't find dependency on target '" + target.getLabel() + "'");
  }
}
