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

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.rules.java.JavaRuleClasses.CONTAINS_JAVA_PROVIDER;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.PackageSpecificationProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/** Rule definition for {@code java_plugin_configuration} */
public class JavaPluginConfigurationRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(java_plugin_configuration).ATTRIBUTE(packages) -->
        The set of <code><a href="${link package_group}">package_group</a></code>s
        the plugins in this configuration should be enabled for.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("packages", BuildType.LABEL_LIST)
                .cfg(HOST)
                .allowedFileTypes()
                .mandatoryNativeProviders(ImmutableList.of(PackageSpecificationProvider.class)))
        /* <!-- #BLAZE_RULE(java_plugin_configuration).ATTRIBUTE(packages) -->
        The list of <code><a href="${link java_plugin}">java_plugin</a></code>s included in this
        configuration.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("plugins", BuildType.LABEL_LIST)
                .cfg(HOST)
                .allowedFileTypes()
                .mandatoryProvidersList(ImmutableList.of(CONTAINS_JAVA_PROVIDER)))
        .add(attr("output_licenses", LICENSE))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("java_plugin_configuration")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(JavaPluginConfiguration.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = java_plugin_configuration, TYPE = OTHER, FAMILY = Java) -->

<p>
Configures a set of <code><a href="${link java_plugin}">java_plugin</a></code>s to run on a set of
packages. Plugin configurations can be added to
<code><a href="${link java_toolchain.plugins}">java_toolchain.plugins</a></code>s.
</p>

<h4 id="java_plugin_configuration">Example:</h4>

<pre class="code">
java_plugin_configuration(
    plugins = [":my_java_plugin"],
    packages = [":plugin_packages"],
)

package_group(
    name = "plugin_packages",
    packages = [
        "//com/my/project/...",
        "-//com/my/project/testing/...",
    ],
)

java_toolchain(
    ...,
    plugin_configuration = [
        ":my_plugin",
    ]
)
</pre>

<!-- #END_BLAZE_RULE -->*/
