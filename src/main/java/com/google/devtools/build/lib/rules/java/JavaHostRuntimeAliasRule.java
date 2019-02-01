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
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.rules.LateBoundAlias.CommonAliasRule;

/** Implementation of the {@code java_runtime_alias} rule. */
public class JavaHostRuntimeAliasRule extends CommonAliasRule<JavaConfiguration> {
  public JavaHostRuntimeAliasRule() {
    super("java_host_runtime_alias", JavaSemantics::hostJdkAttribute, JavaConfiguration.class);
  }

  @Override
  protected Attribute.Builder<Label> makeAttribute(RuleDefinitionEnvironment environment) {
    Attribute.Builder<Label> builder = super.makeAttribute(environment);
    return builder.cfg(HostTransition.INSTANCE).mandatoryProviders(ToolchainInfo.PROVIDER.id());
  }
}
