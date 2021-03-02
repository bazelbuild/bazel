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
import static com.google.devtools.build.lib.packages.RuleClass.Builder.STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME;

import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.ToolchainResolutionMode;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.rules.LateBoundAlias.CommonAliasRule;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;

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
public class LabelBuildSettings {
  @AutoCodec @VisibleForSerialization
  // TODO(b/65746853): find a way to do this without passing the entire BuildConfiguration
  static final LabelLateBoundDefault<BuildConfiguration> ACTUAL =
      LabelLateBoundDefault.fromTargetConfiguration(
          BuildConfiguration.class,
          null,
          (rule, attributes, configuration) -> {
            if (rule == null || configuration == null) {
              return attributes.get(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME, LABEL);
            }
            Object commandLineValue =
                configuration.getOptions().getStarlarkOptions().get(rule.getLabel());
            Label asLabel;
            try {
              asLabel =
                  commandLineValue == null
                      ? attributes.get(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME, LABEL)
                      : LABEL.convert(commandLineValue, "label_flag value resolution");
            } catch (ConversionException e) {
              throw new IllegalStateException(
                  "Getting here means we must have processed a transition via"
                      + " StarlarkTransition.validate, which checks that LABEL.convert works"
                      + " without error.",
                  e);
            }
            return asLabel;
          });

  private static RuleClass buildRuleClass(RuleClass.Builder builder, boolean flag) {
    return builder
        .removeAttribute("licenses")
        .removeAttribute("distribs")
        .add(attr(":alias", LABEL).value(ACTUAL))
        .setBuildSetting(BuildSetting.create(flag, LABEL))
        .canHaveAnyProvider()
        .useToolchainResolution(ToolchainResolutionMode.DISABLED)
        .build();
  }

  /** Rule definition of label_setting */
  public static class LabelBuildSettingRule extends CommonAliasRule<BuildConfiguration> {

    public LabelBuildSettingRule() {
      super("label_setting", env -> ACTUAL, BuildConfiguration.class);
    }

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return buildRuleClass(builder, false);
    }
  }

  /** Rule definition of label_flag */
  public static class LabelBuildFlagRule extends CommonAliasRule<BuildConfiguration> {

    public LabelBuildFlagRule() {
      super("label_flag", env -> ACTUAL, BuildConfiguration.class);
    }

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return buildRuleClass(builder, true);
    }
  }
}
