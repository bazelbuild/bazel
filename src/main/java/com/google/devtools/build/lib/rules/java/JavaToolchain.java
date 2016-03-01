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
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
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
    final NestedSet<Artifact> bootclasspath = getArtifactList("bootclasspath", ruleContext);
    final NestedSet<Artifact> extclasspath = getArtifactList("extclasspath", ruleContext);
    final String encoding = ruleContext.attributes().get("encoding", Type.STRING);
    final List<String> xlint = ruleContext.attributes().get("xlint", Type.STRING_LIST);
    final List<String> misc = ruleContext.attributes().get("misc", Type.STRING_LIST);
    final List<String> jvmOpts = ruleContext.attributes().get("jvm_opts", Type.STRING_LIST);
    Artifact javac = getArtifact("javac", ruleContext);
    Artifact javabuilder = getArtifact("javabuilder", ruleContext);
    Artifact headerCompiler = getArtifact("header_compiler", ruleContext);
    Artifact singleJar = getArtifact("singlejar", ruleContext);
    Artifact genClass = getArtifact("genclass", ruleContext);
    FilesToRunProvider ijar = getExecutable("ijar", ruleContext);
    // TODO(cushon): clean up nulls once migration from --javac_bootclasspath and --javac_extdir
    // is complete, and java_toolchain.{bootclasspath,extclasspath} are mandatory
    final JavaToolchainData toolchainData =
        new JavaToolchainData(
            source,
            target,
            execPathsOrNull(bootclasspath),
            execPathsOrNull(extclasspath),
            encoding,
            xlint,
            misc,
            jvmOpts);
    final JavaConfiguration configuration = ruleContext.getFragment(JavaConfiguration.class);
    JavaToolchainProvider provider =
        new JavaToolchainProvider(
            toolchainData,
            bootclasspath,
            extclasspath,
            configuration.getDefaultJavacFlags(),
            configuration.getDefaultJavaBuilderJvmFlags(),
            javac,
            javabuilder,
            headerCompiler,
            singleJar,
            genClass,
            ijar);
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext)
        .add(JavaToolchainProvider.class, provider)
        .setFilesToBuild(new NestedSetBuilder<Artifact>(Order.STABLE_ORDER).build())
        .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));

    return builder.build();
  }

  /** Returns the exec paths of the given artifacts, or {@code null}. */
  private Iterable<String> execPathsOrNull(NestedSet<Artifact> artifacts) {
    return artifacts != null ? Artifact.toExecPaths(artifacts) : null;
  }

  private FilesToRunProvider getExecutable(String attributeName, RuleContext ruleContext) {
    TransitiveInfoCollection prerequisite = ruleContext.getPrerequisite(attributeName, Mode.HOST);
    if (prerequisite == null) {
      return null;
    }
    return prerequisite.getProvider(FilesToRunProvider.class);
  }

  private Artifact getArtifact(String attributeName, RuleContext ruleContext) {
    TransitiveInfoCollection prerequisite = ruleContext.getPrerequisite(attributeName, Mode.HOST);
    if (prerequisite == null) {
      return null;
    }
    Iterable<Artifact> artifacts = prerequisite.getProvider(FileProvider.class).getFilesToBuild();
    if (Iterables.size(artifacts) != 1) {
      ruleContext.attributeError(
          attributeName, prerequisite.getLabel() + " expected a single artifact");
      return null;
    }
    return Iterables.getOnlyElement(artifacts);
  }

  private NestedSet<Artifact> getArtifactList(String attributeName, RuleContext ruleContext) {
    TransitiveInfoCollection prerequisite = ruleContext.getPrerequisite(attributeName, Mode.HOST);
    if (prerequisite == null) {
      return null;
    }
    return prerequisite.getProvider(FileProvider.class).getFilesToBuild();
  }
}
