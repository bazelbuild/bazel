// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view.config;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.FileProvider;
import com.google.devtools.build.lib.view.FilesToRunProvider;
import com.google.devtools.build.lib.view.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.RunfilesProvider;

import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

import java.util.List;

/**
 * Implementation for the config_setting rule.
 *
 * <p>This is a "pseudo-rule" in that its purpose isn't to generate output artifacts
 * from input artifacts. Rather, it provides configuration context to rules that
 * depend on it.
 */
public class ConfigSetting implements RuleConfiguredTargetFactory {

  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    // Get the required flag=value settings for this rule.
    List<List<String>> settings = NonconfigurableAttributeMapper.of(ruleContext.getRule())
        .get(ConfigRuleClasses.ConfigSettingRule.SETTINGS_ATTRIBUTE, Type.STRING_DICT);
    if (settings.isEmpty()) {
      ruleContext.attributeError(ConfigRuleClasses.ConfigSettingRule.SETTINGS_ATTRIBUTE,
          "no settings specified");
      return null;
    }

    ConfigMatchingProvider configMatcher;
    try {
      final boolean match = matchesConfig(settings, ruleContext.getConfiguration());
      final Label label = ruleContext.getLabel();
      configMatcher = new ConfigMatchingProvider() {
        @Override public boolean matches() { return match; }
        @Override public Label label() { return label; }
      };
    } catch (OptionsParsingException e) {
      ruleContext.attributeError(ConfigRuleClasses.ConfigSettingRule.SETTINGS_ATTRIBUTE,
          "error while parsing configuration settings: " + e.getMessage());
      return null;
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .add(FileProvider.class, new FileProvider(ruleContext.getLabel(),
            NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)))
        .add(FilesToRunProvider.class, new FilesToRunProvider(ruleContext.getLabel(),
            ImmutableList.<Artifact>of(), null, null))
        .add(ConfigMatchingProvider.class, configMatcher)
        .build();
  }

  /**
   * Given a list of [flagName, flagValue] pairs, returns true if flagName == flagValue for
   * every item in the list under this configuration, false otherwise.
   */
  private boolean matchesConfig(List<List<String>> expectedSettings, BuildConfiguration config)
      throws OptionsParsingException {
    for (List<String> setting : expectedSettings) {
      String optionName = setting.get(0);
      Class optionClass = config.getOptionClass(optionName);
      if (optionClass == null) {
        throw new OptionsParsingException("unknown option: '" + optionName + "'");
      }
      String optionValue = setting.get(1);

      OptionsParser parser = OptionsParser.newOptionsParser(optionClass);
      parser.parse("--" + optionName + "=" + optionValue);

      Object expectedValue = parser.getOptions(optionClass).asMap().get(optionName);
      // TODO(bazel-team): support multi-value parameters.
      Object actualValue = config.getOptionValue(optionName);

      if ((expectedValue == null && actualValue != null) || !expectedValue.equals(actualValue)) {
        return false;
      }
    }
    return true;
  }
}
