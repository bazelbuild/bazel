// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Type;

import java.util.List;

/**
 * Implementation for the {@code java_toolchain} rule.
 */
public final class JavaToolchain implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) {
    final String source = ruleContext.attributes().get("source_version", Type.STRING);
    final String target = ruleContext.attributes().get("target_version", Type.STRING);
    final NestedSet<Artifact> bootclasspath = getBootclasspath(ruleContext);
    final NestedSet<Artifact> extclasspath = getExtclasspath(ruleContext);
    final String encoding = ruleContext.attributes().get("encoding", Type.STRING);
    final List<String> xlint = ruleContext.attributes().get("xlint", Type.STRING_LIST);
    final List<String> misc = ruleContext.attributes().get("misc", Type.STRING_LIST);
    final List<String> jvmOpts = ruleContext.attributes().get("jvm_opts", Type.STRING_LIST);
    // TODO(cushon): clean up nulls once migration from --javac_bootclasspath and --javac_extdir
    // is complete, and java_toolchain.{bootclasspath,extclasspath} are mandatory
    final JavaToolchainData toolchainData =
        new JavaToolchainData(
            source,
            target,
            bootclasspath != null ? Artifact.toExecPaths(bootclasspath) : null,
            extclasspath != null ? Artifact.toExecPaths(extclasspath) : null,
            encoding,
            xlint,
            misc,
            jvmOpts);
    final JavaConfiguration configuration = ruleContext.getFragment(JavaConfiguration.class);
    Artifact headerCompiler = getTurbine(ruleContext);
    JavaToolchainProvider provider =
        new JavaToolchainProvider(
            toolchainData,
            bootclasspath,
            extclasspath,
            configuration.getDefaultJavacFlags(),
            configuration.getDefaultJavaBuilderJvmFlags(),
            headerCompiler);
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext)
        .add(JavaToolchainProvider.class, provider)
        .setFilesToBuild(new NestedSetBuilder<Artifact>(Order.STABLE_ORDER).build())
        .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));

    return builder.build();
  }

  private Artifact getTurbine(RuleContext ruleContext) {
    TransitiveInfoCollection prerequisite =
        ruleContext.getPrerequisite("header_compiler", Mode.HOST);
    if (prerequisite == null) {
      return null;
    }
    Iterable<Artifact> artifacts = prerequisite.getProvider(FileProvider.class).getFilesToBuild();
    if (Iterables.size(artifacts) != 1) {
      ruleContext.attributeError(
          "header_compiler", prerequisite.getLabel() + " expected a single artifact");
      return null;
    }
    return Iterables.getOnlyElement(artifacts);
  }

  private NestedSet<Artifact> getBootclasspath(RuleContext ruleContext) {
    TransitiveInfoCollection prerequisite =
        ruleContext.getPrerequisite("bootclasspath", Mode.HOST);
    if (prerequisite == null) {
      return null;
    }
    return prerequisite.getProvider(FileProvider.class).getFilesToBuild();
  }

  private NestedSet<Artifact> getExtclasspath(RuleContext ruleContext) {
    TransitiveInfoCollection prerequisite =
        ruleContext.getPrerequisite("extclasspath", Mode.HOST);
    if (prerequisite == null) {
      return null;
    }
    return prerequisite.getProvider(FileProvider.class).getFilesToBuild();
  }
}
