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

import static com.google.devtools.build.lib.rules.java.JavaSemantics.JAVA_TOOLCHAIN_LABEL;

import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.rules.LateBoundAlias.CommonAliasRule;
import java.io.Serializable;

/** Implementation of the {@code java_toolchain_alias} rule. */
public class JavaToolchainAliasRule extends CommonAliasRule<JavaConfiguration> {
  public JavaToolchainAliasRule() {
    super(
        "java_toolchain_alias",
        JavaToolchainAliasRule::javaToolchainAttribute,
        JavaConfiguration.class);
  }

  static LabelLateBoundDefault<JavaConfiguration> javaToolchainAttribute(
      RuleDefinitionEnvironment environment) {
    return LabelLateBoundDefault.fromTargetConfiguration(
        JavaConfiguration.class,
        environment.getToolsLabel(JAVA_TOOLCHAIN_LABEL),
        (Attribute.LateBoundDefault.Resolver<JavaConfiguration, Label> & Serializable)
            (rule, attributes, javaConfig) -> javaConfig.getToolchainLabel());
  }
}
