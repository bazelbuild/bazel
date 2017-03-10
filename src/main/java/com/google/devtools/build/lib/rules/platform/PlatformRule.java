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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for {@link Platform}. */
public class PlatformRule implements RuleDefinition {
  public static final String RULE_NAME = "platform";
  public static final String CONSTRAINT_VALUES_ATTR = "constraint_values";
  public static final String REMOTE_EXECUTION_PROPS_ATTR = "remote_execution_properties";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    /* <!-- #BLAZE_RULE(platform).NAME -->
    <!-- #END_BLAZE_RULE.NAME --> */
    return builder
        .override(
            attr("tags", Type.STRING_LIST)
                // No need to show up in ":all", etc. target patterns.
                .value(ImmutableList.of("manual"))
                .nonconfigurable("low-level attribute, used in platform configuration"))
        .add(
            attr(CONSTRAINT_VALUES_ATTR, BuildType.LABEL_LIST)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .mandatoryNativeProviders(
                    ImmutableList.<Class<? extends TransitiveInfoProvider>>of(
                        ConstraintValueProvider.class)))
        .add(attr(REMOTE_EXECUTION_PROPS_ATTR, Type.STRING_DICT))
        .removeAttribute("deps")
        .removeAttribute("data")
        .exemptFromConstraintChecking("this rule is part of constraint definition")
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return Metadata.builder()
        .name(RULE_NAME)
        .ancestors(BaseRuleClasses.RuleBase.class)
        .factoryClass(Platform.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = platform, TYPE = OTHER, FAMILY = Platform)[GENERIC_RULE] -->

<p>This rule defines a platform, as a collection of constraint_values.

<h4 id="platform_examples">Examples</h4>
<p>
  This defines two possible platforms, each targeting a different CPU type.
</p>
<pre class="code">
constraint_setting(name="cpu")
constraint_value(
    name="arm64",
    constraint=":cpu")
constraint_value(
    name="k8",
    constraint=":cpu")
platform(
    name="mobile_device",
    constraints = [
        ":arm64",
    ])
platform(
    name="devel",
    constraints = [
        ":k8",
    ])
</pre>

<!-- #END_BLAZE_RULE -->*/
