// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.rules.java;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

import java.util.List;

/**
 * Implementation for the {@code java_toolchain} rule.
 */
public final class JavaToolchain implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) {
    final String source = ruleContext.attributes().get("source_version", Type.STRING);
    final String target = ruleContext.attributes().get("target_version", Type.STRING);
    final String encoding = ruleContext.attributes().get("encoding", Type.STRING);
    final List<String> xlint = ruleContext.attributes().get("xlint", Type.STRING_LIST);
    final List<String> misc = ruleContext.attributes().get("misc", Type.STRING_LIST);
    final List<String> jvmOpts = ruleContext.attributes().get("jvm_opts", Type.STRING_LIST);
    final JavaToolchainData toolchainData =
        new JavaToolchainData(source, target, encoding, xlint, misc, jvmOpts);
    final JavaConfiguration configuration = ruleContext.getFragment(JavaConfiguration.class);
    JavaToolchainProvider provider = new JavaToolchainProvider(toolchainData,
        configuration.getDefaultJavacFlags(), configuration.getDefaultJavaBuilderJvmFlags());
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext)
        .add(JavaToolchainProvider.class, provider)
        .setFilesToBuild(new NestedSetBuilder<Artifact>(Order.STABLE_ORDER).build())
        .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));

    return builder.build();
  }
}
