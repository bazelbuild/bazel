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

package com.google.devtools.build.lib.analysis.config;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.syntax.Type.STRING_DICT;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.syntax.Type;

import java.util.List;
import java.util.Map;

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
  public static final class ConfigBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .override(attr("tags", Type.STRING_LIST)
               // No need to show up in ":all", etc. target patterns.
              .value(ImmutableList.of("manual"))
              .nonconfigurable(NONCONFIGURABLE_ATTRIBUTE_REASON))
          .exemptFromConstraintChecking(
              "these rules don't include content that gets built into their dependers")
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$config_base_rule")
          .type(RuleClass.Builder.RuleClassType.ABSTRACT)
          .ancestors(BaseRuleClasses.BaseRule.class)
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
  public static final class ConfigSettingRule implements RuleDefinition {
    /**
     * The name of this rule.
     */
    public static final String RULE_NAME = "config_setting";

    /**
     * The name of the attribute that declares flag bindings.
     */
    public static final String SETTINGS_ATTRIBUTE = "values";

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE(config_setting).ATTRIBUTE(values) -->
          The set of configuration values that match this rule (expressed as Blaze flags)
          ${SYNOPSIS}

          <i>(Dictionary mapping flags to expected values, both expressed as strings;
             mandatory)</i>

          <p>This rule inherits the configuration of the configured target that
            references it in a <code>select</code> statement. It is considered to
            "match" a Blaze invocation if, for every entry in the dictionary, its
            configuration matches the entry's expected value. For example
            <code>values = {"compilation_mode": "opt"}</code> matches the invocations
            <code>blaze build --compilation_mode=opt ...</code> and
            <code>blaze build -c opt ...</code> on target-configured rules.
          </p>

          <p>For convenience's sake, configuration values are specified as Blaze flags (without
            the preceding <code>"--"</code>). But keep in mind that the two are not the same. This
            is because targets can be built in multiple configurations within the same
            build. For example, a host configuration's "cpu" matches the value of
            <code>--host_cpu</code>, not <code>--cpu</code>. So different instances of the
            same <code>config_setting</code> may match the same invocation differently
            depending on the configuration of the rule using them.
          </p>

          <p>If a flag is not explicitly set at the command line, its default value is used.
             If a key appears multiple times in the dictionary, only the last instance is used.
             If a key references a flag that can be set multiple times on the command line (e.g.
             <code>blaze build --copt=foo --copt=bar --copt=baz ...</code>), a match occurs if
             <i>any</i> of those settings match.
          <p>

          <p>This attribute cannot be empty.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr(SETTINGS_ATTRIBUTE, STRING_DICT).mandatory()
              .nonconfigurable(NONCONFIGURABLE_ATTRIBUTE_REASON))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name(RULE_NAME)
          .type(RuleClass.Builder.RuleClassType.NORMAL)
          .ancestors(ConfigBaseRule.class)
          .factoryClass(ConfigSetting.class)
          .build();
    }

    /**
     * config_setting can't use {@link RuleClass.Builder#requiresConfigurationFragments} because
     * config_setting's dependencies come from option names as strings. This special override
     * computes that properly.
     */
    public static List<Class<? extends BuildConfiguration.Fragment>> requiresConfigurationFragments(
        Rule rule, Map<String, Class<? extends BuildConfiguration.Fragment>> optionsToFragmentMap) {
      ImmutableList.Builder<Class<? extends BuildConfiguration.Fragment>> builder =
          ImmutableList.builder();
      AttributeMap attributes = NonconfigurableAttributeMapper.of(rule);
      for (String optionName : attributes.get(SETTINGS_ATTRIBUTE, Type.STRING_DICT).keySet()) {
        if (optionName.equals("cpu")) {
          // The "cpu" flag is special: it's defined in BuildConfiguration.Options but its value
          // is set in CppConfiguration (which reads a CROSSTOOL to determine that value).
          // So this requires a special mapping.
          builder.add(getCppConfiguration(optionsToFragmentMap.values()));
        } else {
          Class<? extends BuildConfiguration.Fragment> value = optionsToFragmentMap.get(optionName);
          // Null values come from BuildConfiguration.Options, which is implicitly included.
          if (value != null) {
            builder.add(value);
          }
        }
      }
      return builder.build();
    }

    /**
     * We can't directly reference CppConfiguration.class because it's in a different Bazel library.
     * While we could add that library as a dep, that would bring in a bunch of unnecessary C++ and
     * crosstool code to what's otherwise a language-agnostic library. So we use a bit of
     * introspection instead.
     */
    private static Class<? extends BuildConfiguration.Fragment> getCppConfiguration(
        Iterable<Class<? extends BuildConfiguration.Fragment>> configs) {
      for (Class<? extends BuildConfiguration.Fragment> clazz : configs) {
        if (clazz.getSimpleName().equals("CppConfiguration")) {
          return clazz;
        }
      }
      throw new IllegalStateException("Couldn't find the C++ fragment");
    }
  }

/*<!-- #BLAZE_RULE (NAME = config_setting, TYPE = OTHER, FAMILY = General)[GENERIC_RULE] -->

<p>
  Matches an expected configuration state (expressed as Blaze flags) for the purpose of triggering
  configurable attributes. See <a href="functions.html#select">select</a> for how to consume this
  rule and <a href="common-definitions.html#configurable-attributes">
  Configurable attributes</a> for an overview of the general feature.

<h4 id="config_setting_examples">Examples</h4>

<p>The following matches any Blaze invocation that specifies <code>--compilation_mode=opt</code>
   or <code>-c opt</code> (either explicitly at the command line or implicitly from .blazerc
   files, etc.), when applied to a target configuration rule:
</p>

<pre class="code">
config_setting(
    name = "simple",
    values = {"compilation_mode": "opt"}
)
</pre>

<p>The following matches any Blaze invocation that builds for ARM and applies a custom define
   (e.g. <code>blaze build --cpu=armeabi --define FOO=bar ...</code>), , when applied to a target
   configuration rule:
</p>

<pre class="code">
config_setting(
    name = "two_conditions",
    values = {
        "cpu": "armeabi",
        "define": "FOO=bar"
    }
)
</pre>

<h4 id="config_setting_notes">Notes</h4>

<p>See <a href="functions.html#select">select</a> for policies on what happens depending on how
   many rules match an invocation.
</p>

<p>For flags that support shorthand forms (e.g. <code>--compilation_mode</code> vs.
  <code>-c</code>), <code>values</code> definitions must use the full form. These automatically
  match invocations using either form.
</p>

<p>The currently endorsed method for creating custom conditions that can't be expressed through
  dedicated build flags is through the --define flag. Use this flag with caution: it's not ideal
  and only endorsed for lack of a currently better workaround. See the
  <a href="common-definitions.html#configurable-attributes">
  Configurable attributes</a> section for more discussion.
</p>

<p>Try to consolidate <code>config_setting</code> definitions as much as possible. In other words,
  define <code>//common/conditions:foo</code> in one common package instead of repeating separate
  instances in <code>//project1:foo</code>, <code>//project2:foo</code>, etc. that all mean the
  same thing.
</p>

<!-- #END_BLAZE_RULE -->*/
}
