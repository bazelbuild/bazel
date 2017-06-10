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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.LocationExpander;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
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
    final String source = ruleContext.attributes().get("source_version", Type.STRING);
    final String target = ruleContext.attributes().get("target_version", Type.STRING);
    final NestedSet<Artifact> bootclasspath = getArtifactList("bootclasspath", ruleContext);
    final NestedSet<Artifact> extclasspath = getArtifactList("extclasspath", ruleContext);
    final String encoding = ruleContext.attributes().get("encoding", Type.STRING);
    final List<String> xlint = ruleContext.attributes().get("xlint", Type.STRING_LIST);
    final List<String> misc = ruleContext.getTokenizedStringListAttr("misc");
    final boolean javacSupportsWorkers =
        ruleContext.attributes().get("javac_supports_workers", Type.BOOLEAN);
    TransitiveInfoCollection javacDep = ruleContext.getPrerequisite("javac", Mode.HOST);
    Artifact javac = null;
    NestedSet<Artifact> javacJars = javacDep.getProvider(FileProvider.class).getFilesToBuild();
    if (Iterables.size(javacJars) == 1) {
      javac = Iterables.getOnlyElement(javacJars);
    } else {
      ruleContext.attributeError("javac", javacDep.getLabel() + " expected a single artifact");
      return null;
    }
    final List<String> jvmOpts =
        getJvmOpts(
            ruleContext,
            ImmutableMap.<Label, ImmutableCollection<Artifact>>of(
                javacDep.getLabel(), ImmutableList.of(javac)));
    Artifact javabuilder = getArtifact("javabuilder", ruleContext);
    Artifact headerCompiler = getArtifact("header_compiler", ruleContext);
    boolean forciblyDisableHeaderCompilation =
        ruleContext.attributes().get("forcibly_disable_header_compilation", Type.BOOLEAN);
    Artifact singleJar = getArtifact("singlejar", ruleContext);
    Artifact oneVersion = getArtifact("oneversion", ruleContext);
    Artifact oneVersionWhitelist = getArtifact("oneversion_whitelist", ruleContext);
    Artifact genClass = getArtifact("genclass", ruleContext);
    Artifact resourceJarBuilder = getArtifact("resourcejar", ruleContext);
    Artifact timezoneData = getArtifact("timezone_data", ruleContext);
    FilesToRunProvider ijar = ruleContext.getExecutablePrerequisite("ijar", Mode.HOST);
    ImmutableListMultimap<String, String> compatibleJavacOptions =
        getCompatibleJavacOptions(ruleContext);

    final JavaToolchainData toolchainData =
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
    final JavaConfiguration configuration = ruleContext.getFragment(JavaConfiguration.class);
    JavaToolchainProvider provider =
        JavaToolchainProvider.create(
            ruleContext.getLabel(),
            toolchainData,
            bootclasspath,
            extclasspath,
            configuration.getDefaultJavacFlags(),
            javac,
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
            .add(JavaToolchainProvider.class, provider)
            .setFilesToBuild(new NestedSetBuilder<Artifact>(Order.STABLE_ORDER).build())
            .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));

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

  private static Artifact getArtifact(String attributeName, RuleContext ruleContext) {
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

  private static NestedSet<Artifact> getArtifactList(
      String attributeName, RuleContext ruleContext) {
    TransitiveInfoCollection prerequisite = ruleContext.getPrerequisite(attributeName, Mode.HOST);
    if (prerequisite == null) {
      return null;
    }
    return prerequisite.getProvider(FileProvider.class).getFilesToBuild();
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
