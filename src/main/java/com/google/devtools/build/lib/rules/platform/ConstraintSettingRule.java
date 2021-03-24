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
// limitations under the License.

package com.google.devtools.build.lib.rules.platform;

import static com.google.devtools.build.lib.packages.Attribute.attr;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;

/** Rule definition for {@link ConstraintSetting}. */
public class ConstraintSettingRule implements RuleDefinition {
  public static final String RULE_NAME = "constraint_setting";
  public static final String DEFAULT_CONSTRAINT_VALUE_ATTR = "default_constraint_value";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .advertiseStarlarkProvider(ConstraintSettingInfo.PROVIDER.id())
        /* <!-- #BLAZE_RULE(constraint_setting).ATTRIBUTE(default_constraint_value) -->
        The label of the default value for this setting, to be used if no value is given. If this
        attribute is present, the <code>constraint_value</code> it points to must be defined in the
        same package as this <code>constraint_setting</code>.

        <p>If a constraint setting has a default value, then whenever a platform does not include
        any constraint value for that setting, it is the same as if the platform had specified the
        default value. Otherwise, if there is no default value, the constraint setting is considered
        to be unspecified by that platform. In that case, the platform would not match against any
        constraint list (such as for a <code>config_setting</code>) that requires a particular value
        for that setting.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr(DEFAULT_CONSTRAINT_VALUE_ATTR, BuildType.NODEP_LABEL))
        .build();
  }

  @Override
  public RuleDefinition.Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(RULE_NAME)
        .ancestors(PlatformBaseRule.class)
        .factoryClass(ConstraintSetting.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = constraint_setting, FAMILY = Platform)[GENERIC_RULE] -->

<p>This rule is used to introduce a new constraint type for which a platform may specify a value.
For instance, you might define a <code>constraint_setting</code> named "glibc_version" to represent
the capability for platforms to have different versions of the glibc library installed.

For more details, see the
<a href="https://docs.bazel.build/versions/master/platforms.html">Platforms</a> page.

<p>Each <code>constraint_setting</code> has an extensible set of associated
<code>constraint_value</code>s. Usually these are defined in the same package, but sometimes a
different package will introduce new values for an existing setting. For instance, the predefined
setting <code>@platforms//cpu:cpu</code> can be extended with a custom value in order to
define a platform targeting an obscure cpu architecture.

<!-- #END_BLAZE_RULE -->*/
