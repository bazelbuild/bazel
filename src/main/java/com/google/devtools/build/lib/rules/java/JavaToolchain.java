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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.LocationExpander;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.java.JavaToolchainData.SupportsWorkers;
import com.google.devtools.build.lib.syntax.Type;
import java.util.List;
import java.util.Map;

/**
 * Implementation for the {@code java_toolchain} rule.
 */
public final class JavaToolchain implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws RuleErrorException {
    String source = ruleContext.attributes().get("source_version", Type.STRING);
    String target = ruleContext.attributes().get("target_version", Type.STRING);
    NestedSet<Artifact> bootclasspath = PrerequisiteArtifacts.nestedSet(
        ruleContext, "bootclasspath", Mode.HOST);
    NestedSet<Artifact> extclasspath = PrerequisiteArtifacts.nestedSet(
        ruleContext, "extclasspath", Mode.HOST);
    String encoding = ruleContext.attributes().get("encoding", Type.STRING);
    List<String> xlint = ruleContext.attributes().get("xlint", Type.STRING_LIST);
    List<String> misc = ruleContext.getExpander().withDataLocations().tokenized("misc");
    boolean javacSupportsWorkers =
        ruleContext.attributes().get("javac_supports_workers", Type.BOOLEAN);
    Artifact javac = ruleContext.getPrerequisiteArtifact("javac", Mode.HOST);
    Artifact javabuilder = ruleContext.getPrerequisiteArtifact("javabuilder", Mode.HOST);
    Artifact headerCompiler = ruleContext.getPrerequisiteArtifact("header_compiler", Mode.HOST);
    boolean forciblyDisableHeaderCompilation =
        ruleContext.attributes().get("forcibly_disable_header_compilation", Type.BOOLEAN);
    Artifact singleJar = ruleContext.getPrerequisiteArtifact("singlejar", Mode.HOST);
    Artifact oneVersion = ruleContext.getPrerequisiteArtifact("oneversion", Mode.HOST);
    Artifact oneVersionWhitelist = ruleContext
        .getPrerequisiteArtifact("oneversion_whitelist", Mode.HOST);
    Artifact genClass = ruleContext.getPrerequisiteArtifact("genclass", Mode.HOST);
    Artifact resourceJarBuilder = ruleContext.getPrerequisiteArtifact("resourcejar", Mode.HOST);
    Artifact timezoneData = ruleContext.getPrerequisiteArtifact("timezone_data", Mode.HOST);
    FilesToRunProvider ijar = ruleContext.getExecutablePrerequisite("ijar", Mode.HOST);
    ImmutableListMultimap<String, String> compatibleJavacOptions =
        getCompatibleJavacOptions(ruleContext);

    NestedSet<Artifact> tools = PrerequisiteArtifacts.nestedSet(ruleContext, "tools", Mode.HOST);

    TransitiveInfoCollection javacDep = ruleContext.getPrerequisite("javac", Mode.HOST);
    List<String> jvmOpts =
        getJvmOpts(
            ruleContext,
            ImmutableMap.<Label, ImmutableCollection<Artifact>>of(
                AliasProvider.getDependencyLabel(javacDep), ImmutableList.of(javac)));

    JavaToolchainData toolchainData =
        new JavaToolchainData(
            source,
            target,
            Artifact.toExecPaths(bootclasspath),
            Artifact.toExecPaths(extclasspath),
            encoding,
            xlint,
            misc,
            jvmOpts,
            javacSupportsWorkers ? SupportsWorkers.YES : SupportsWorkers.NO);
    JavaConfiguration configuration = ruleContext.getFragment(JavaConfiguration.class);
    JavaToolchainProvider provider =
        JavaToolchainProvider.create(
            ruleContext.getLabel(),
            toolchainData,
            bootclasspath,
            extclasspath,
            configuration.getDefaultJavacFlags(),
            javac,
            tools,
            javabuilder,
            headerCompiler,
            forciblyDisableHeaderCompilation,
            singleJar,
            oneVersion,
            oneVersionWhitelist,
            genClass,
            resourceJarBuilder,
            timezoneData,
            ijar,
            compatibleJavacOptions);
    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext)
            .addSkylarkTransitiveInfo(
                JavaToolchainSkylarkApiProvider.NAME, new JavaToolchainSkylarkApiProvider())
            .addProvider(JavaToolchainProvider.class, provider)
            .addProvider(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
            .setFilesToBuild(new NestedSetBuilder<Artifact>(Order.STABLE_ORDER).build());

    return builder.build();
  }

  private static ImmutableListMultimap<String, String> getCompatibleJavacOptions(
      RuleContext ruleContext) {
    ImmutableListMultimap.Builder<String, String> result = ImmutableListMultimap.builder();
    for (Map.Entry<String, List<String>> entry :
        ruleContext.attributes().get("compatible_javacopts", Type.STRING_LIST_DICT).entrySet()) {
      result.putAll(entry.getKey(), JavaHelper.tokenizeJavaOptions(entry.getValue()));
    }
    return result.build();
  }

  private static ImmutableList<String> getJvmOpts(
      RuleContext ruleContext, ImmutableMap<Label, ImmutableCollection<Artifact>> locations) {
    // LocationExpander is used directly instead of e.g. getExpandedStringListAttr because the
    // latter hard-codes list of attributes that can provide prerequisites.
    LocationExpander expander =
        new LocationExpander(ruleContext, locations, /*allowDataAttributeEntriesInLabel=*/ false);
    ImmutableList.Builder<String> result = ImmutableList.builder();
    for (String option : ruleContext.attributes().get("jvm_opts", Type.STRING_LIST)) {
      result.add(expander.expand(option));
    }
    return result.build();
  }
}
