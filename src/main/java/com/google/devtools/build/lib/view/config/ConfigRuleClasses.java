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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.STRING_DICT;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.view.BaseRuleClasses;
import com.google.devtools.build.lib.view.BlazeRule;
import com.google.devtools.build.lib.view.RuleDefinition;
import com.google.devtools.build.lib.view.RuleDefinitionEnvironment;

/**
 * Definitions for rule classes that specify or manipulate configuration settings.
 *
 * <p>These are not "traditional" rule classes in that they can't be requested as top-level
 * targets and don't translate input artifacts into output artifacts. Instead, they affect
 * how *other* rules work. See individual class comments for details.
 */
public class ConfigRuleClasses {

  private static final String NONCONFIGURABLE_ATTRIBUTE_REASON =
      "part of a rule class that *triggers* configurable behavior";

  /**
   * Common settings for all configurability rules.
   */
  @BlazeRule(name = "$config_base_rule",
               type = RuleClass.Builder.RuleClassType.ABSTRACT,
               ancestors = { BaseRuleClasses.BaseRule.class })
  public static final class ConfigBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .override(attr("tags", Type.STRING_LIST)
              .undocumented("expected of all rules but not relevant for the user here")
               // No need to show up in ":all", etc. target patterns.
              .value(ImmutableList.of("manual"))
              .nonconfigurable(NONCONFIGURABLE_ATTRIBUTE_REASON))
          .build();
    }
  }

  /**
   * A named "partial configuration setting" that specifies a set of command-line
   * "flag=value" bindings.
   *
   * <p>For example:
   * <pre>
   *   config_setting(
   *       name = 'foo',
   *       values = {
   *           'flag1': 'aValue'
   *           'flag2': 'bValue'
   *       })
   * </pre>
   *
   * <p>declares a setting that binds command-line flag <pre>flag1</pre> to value
   * <pre>aValue</pre> and <pre>flag2</pre> to <pre>bValue</pre>.
   *
   * <p>This is used by configurable attributes to determine which branch to
   * follow based on which <pre>config_setting</pre> instance matches all its
   * flag values in the configurable attribute owner's configuration.
   *
   * <p>This rule isn't accessed through the standard {@link RuleContext#getPrerequisites}
   * interface. This is because Bazel constructs a rule's configured attribute map *before*
   * its {@link RuleContext} is created (in fact, the map is an input to the context's
   * constructor). And the config_settings referenced by the rule's configurable attributes are
   * themselves inputs to that map. So Bazel has special logic to read and properly apply
   * config_setting instances. See {@link ConfiguredTargetFunction#getConfigConditions} for details.
   */
  @BlazeRule(name = "config_setting",
               type = RuleClass.Builder.RuleClassType.NORMAL,
               ancestors = { ConfigBaseRule.class },
               factoryClass = ConfigSetting.class)
  public static final class ConfigSettingRule implements RuleDefinition {
    /**
     * The name of the attribute that declares flag bindings.
     */
    public static final String SETTINGS_ATTRIBUTE = "values";

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr(SETTINGS_ATTRIBUTE, STRING_DICT).mandatory()
              .nonconfigurable(NONCONFIGURABLE_ATTRIBUTE_REASON))
          // Keep this rule undocumented while it's under soft (non-announced) release:
          .setUndocumented()
          .build();
    }
  }
}
