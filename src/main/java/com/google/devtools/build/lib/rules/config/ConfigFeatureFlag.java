// Copyright 2017 The Bazel Authors. All rights reserved.
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
// limitations under the License

package com.google.devtools.build.lib.rules.config;

import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Printer;
import java.util.List;

/**
 * The implementation of the config_feature_flag rule for defining custom flags for Android rules.
 */
public class ConfigFeatureFlag implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    List<String> specifiedValues = ruleContext.attributes().get("allowed_values", STRING_LIST);
    ImmutableSet<String> values = ImmutableSet.copyOf(specifiedValues);
    Predicate<String> isValidValue = Predicates.in(values);
    if (values.size() != specifiedValues.size()) {
      ImmutableMultiset<String> groupedValues = ImmutableMultiset.copyOf(specifiedValues);
      ImmutableList.Builder<String> duplicates = new ImmutableList.Builder<String>();
      for (Multiset.Entry<String> value : groupedValues.entrySet()) {
        if (value.getCount() > 1) {
          duplicates.add(value.getElement());
        }
      }
      ruleContext.attributeError(
          "allowed_values",
          "cannot contain duplicates, but contained multiple of "
              + Printer.repr(duplicates.build()));
    }

    String defaultValue = ruleContext.attributes().get("default_value", STRING);
    if (!isValidValue.apply(defaultValue)) {
      ruleContext.attributeError(
          "default_value",
          "must be one of "
              + Printer.repr(values.asList())
              + ", but was "
              + Printer.repr(defaultValue));
    }

    if (ruleContext.hasErrors()) {
      // Don't bother validating the value if the flag was already incorrectly specified without
      // looking at the value.
      return null;
    }

    String value =
        ruleContext
            .getFragment(ConfigFeatureFlagConfiguration.class)
            .getFeatureFlagValue(ruleContext.getOwner())
            .or(defaultValue);

    if (!isValidValue.apply(value)) {
      // TODO(mstaib): When configurationError is available, use that instead.
      ruleContext.ruleError(
          "value must be one of "
              + Printer.repr(values.asList())
              + ", but was "
              + Printer.repr(value));
      return null;
    }

    ConfigFeatureFlagProvider provider = ConfigFeatureFlagProvider.create(value, isValidValue);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER))
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(ConfigFeatureFlagProvider.class, provider)
        .addNativeDeclaredProvider(provider)
        .build();
  }
}
