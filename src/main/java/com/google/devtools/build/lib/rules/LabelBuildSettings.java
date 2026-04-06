// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;
import static com.google.devtools.build.lib.packages.RuleClass.Builder.STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.ToolchainResolutionMode;
import com.google.devtools.build.lib.rules.LateBoundAlias.AbstractAliasRule;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import net.starlark.java.eval.Starlark;

/**
 * Native implementation of label setting and flags.
 *
 * <p>While most build settings are completely defined in starlark, we're natively defining
 * label-typed ones because:
 *
 * <ul>
 *   <li>they're essentially special Alias targets
 *   <li>we don't have a known use case where you'd want to manipulate a label-typed build setting
 *       in its implementation section.
 * </ul>
 *
 * <p>Once we do have (2), we can consider switching over to starlark implementation. The dangers
 * there involve the implementation function returning a label we've never seen before in the build.
 * And since label-typed build settings actually return the providers of the targets they point to,
 * we'd have to be able to load and configure potentially arbitrary labels on the fly. This is not
 * possible today and could easily introduce large performance issues.
 */
public final class LabelBuildSettings {
  @SerializationConstant @VisibleForSerialization
  static final String NONCONFIGURABLE_ATTRIBUTE_REASON =
      "part of a rule class that *triggers* configurable behavior";

  // TODO(b/65746853): find a way to do this without passing the entire BuildConfigurationValue
  static final LabelLateBoundDefault<BuildConfigurationValue> ACTUAL =
      LabelLateBoundDefault.fromTargetConfigurationWithRuleBasedDefault(
          BuildConfigurationValue.class,
          (rule) ->
              // RawAttributeMapper means this attribute can't be select()able (which it isn't).
              RawAttributeMapper.of(rule)
                  .get(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME, NODEP_LABEL),
          (rule, attributes, configuration) -> {
            if (rule == null || configuration == null) {
              return attributes.get(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME, NODEP_LABEL);
            }
            Object commandLineValue =
                configuration.getOptions().getStarlarkOptions().get(rule.getLabel());
            if (commandLineValue == null) {
              return attributes.get(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME, NODEP_LABEL);
            }
            Preconditions.checkState(
                commandLineValue instanceof Label,
                "the value of %s should have been converted to a label already, but its type is %s",
                rule.getLabel(),
                Starlark.type(commandLineValue));
            return (Label) commandLineValue;
          });

  private static RuleClass buildRuleClass(RuleClass.Builder builder, boolean flag) {
    return builder
        .removeAttribute("licenses")
        .removeAttribute("distribs")
        .removeAttribute(":action_listener")
        .add(attr(":alias", LABEL).value(ACTUAL))
        .add(
            attr("scope", STRING)
                .value("universal")
                .nonconfigurable(NONCONFIGURABLE_ATTRIBUTE_REASON))
        .add(attr("on_leave_scope", NODEP_LABEL).nonconfigurable(NONCONFIGURABLE_ATTRIBUTE_REASON))
        /* <!-- #BLAZE_RULE(label_flag).ATTRIBUTE(build_setting_default) -->
        The default label for this build setting's value. The providers of the referenced target
        are forwarded by the <code>label_flag</code> until the value is changed by the command line
        or by a transition.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        /* <!-- #BLAZE_RULE(label_setting).ATTRIBUTE(build_setting_default) -->
        The default label for this build setting's value. The providers of the referenced target
        are forwarded by the <code>label_setting</code> until the value is changed by a transition.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .setBuildSetting(BuildSetting.create(flag, NODEP_LABEL))
        .canHaveAnyProvider()
        .toolchainResolutionMode(ToolchainResolutionMode.DISABLED)
        .build();
  }

  /** Rule definition of label_setting. */
  public static final class LabelBuildSettingRule extends AbstractAliasRule {

    public LabelBuildSettingRule() {
      super("label_setting");
    }

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return buildRuleClass(builder, /* flag= */ false);
    }
  }

  /** Rule definition of label_flag */
  public static final class LabelBuildFlagRule extends AbstractAliasRule {

    public LabelBuildFlagRule() {
      super("label_flag");
    }

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return buildRuleClass(builder, /* flag= */ true);
    }
  }

  private LabelBuildSettings() {}
}

/*<!-- #BLAZE_RULE (NAME = label_flag, FAMILY = General)[GENERIC_RULE] -->

<p>
  A <code>label_flag</code> is a label-typed build setting that forwards the providers of the
  target referenced by its current value.
</p>

<p>
  Use <code>label_flag</code> when you want a label-typed build setting that can be changed by
  transitions and by users at the command line.
</p>

<h4 id="label_flag_examples">Example</h4>

<pre class="code">
label_flag(
    name = "my_flag",
    build_setting_default = ":default_target",
)

cc_library(
    name = "default_target",
    srcs = ["default.cc"],
)

cc_library(
    name = "other_target",
    srcs = ["other.cc"],
)
</pre>

<p>
  Users can override this with a command like
  <code>bazel build //example:target --//example:my_flag=//example:other_target</code>.
</p>

<!-- #END_BLAZE_RULE -->*/

/*<!-- #BLAZE_RULE (NAME = label_setting, FAMILY = General)[GENERIC_RULE] -->

<p>
  A <code>label_setting</code> is a label-typed build setting that forwards the providers of the
  target referenced by its current value.
</p>

<p>
  Use <code>label_setting</code> for internal configuration that should be changed by transitions
  but not set directly by users on the command line.
</p>

<h4 id="label_setting_examples">Example</h4>

<pre class="code">
label_setting(
    name = "current_tool",
    build_setting_default = ":default_tool",
)

filegroup(
    name = "default_tool",
    srcs = ["tool.txt"],
)
</pre>

<!-- #END_BLAZE_RULE -->*/
