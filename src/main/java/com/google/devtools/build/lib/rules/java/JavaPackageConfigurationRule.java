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

package com.google.devtools.build.lib.rules.java;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.PackageSpecificationProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigAwareRuleClassBuilder;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for {@code java_package_configuration} */
public class JavaPackageConfigurationRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return ConfigAwareRuleClassBuilder.of(builder)
        .requiresHostConfigurationFragments(JavaConfiguration.class)
        .originalBuilder()
        .requiresConfigurationFragments(JavaConfiguration.class)
        /* <!-- #BLAZE_RULE(java_package_configuration).ATTRIBUTE(packages) -->
        The set of <code><a href="${link package_group}">package_group</a></code>s
        the configuration should be applied to.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("packages", LABEL_LIST)
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes()
                .mandatoryBuiltinProviders(ImmutableList.of(PackageSpecificationProvider.class)))
        /* <!-- #BLAZE_RULE(java_package_configuration).ATTRIBUTE(javacopts) -->
        Java compiler flags.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("javacopts", Type.STRING_LIST))
        /* <!-- #BLAZE_RULE(java_package_configuration).ATTRIBUTE(data) -->
        The list of files needed by this configuration at runtime.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("data", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE).dontCheckConstraints())
        .add(attr("output_licenses", LICENSE))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("java_package_configuration")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(JavaPackageConfiguration.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = java_package_configuration, TYPE = OTHER, FAMILY = Java) -->

<p>
Configuration to apply to a set of packages.
Configurations can be added to
<code><a href="${link java_toolchain.javacopts}">java_toolchain.javacopts</a></code>s.
</p>

<h4 id="java_package_configuration_example">Example:</h4>

<pre class="code">
java_package_configuration(
    name = "my_configuration",
    packages = [":my_packages"],
    javacopts = ["-Werror"],
)

package_group(
    name = "my_packages",
    packages = [
        "//com/my/project/...",
        "-//com/my/project/testing/...",
    ],
)

java_toolchain(
    ...,
    package_configuration = [
        ":my_configuration",
    ]
)
</pre>

<!-- #END_BLAZE_RULE -->*/
