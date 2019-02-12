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

import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.rules.LateBoundAlias.CommonAliasRule;
import java.io.Serializable;

/** Implementation of the {@code java_runtime_alias} rule. */
public class JavaRuntimeAliasRule extends CommonAliasRule<JavaConfiguration> {
  public JavaRuntimeAliasRule() {
    super("java_runtime_alias", JavaRuntimeAliasRule::jvmAttribute, JavaConfiguration.class);
  }

  @Override
  protected Attribute.Builder<Label> makeAttribute(RuleDefinitionEnvironment environment) {
    Attribute.Builder<Label> builder = super.makeAttribute(environment);
    return builder.mandatoryProviders(ToolchainInfo.PROVIDER.id());
  }

  static LabelLateBoundDefault<JavaConfiguration> jvmAttribute(RuleDefinitionEnvironment env) {
    return LabelLateBoundDefault.fromTargetConfiguration(
        JavaConfiguration.class,
        env.getToolsLabel(JavaImplicitAttributes.JDK_LABEL),
        (Attribute.LateBoundDefault.Resolver<JavaConfiguration, Label> & Serializable)
            (rule, attributes, configuration) -> configuration.getRuntimeLabel());
  }
}
