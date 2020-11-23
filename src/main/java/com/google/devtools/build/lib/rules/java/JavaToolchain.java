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

import static com.google.common.base.Strings.isNullOrEmpty;
import static com.google.common.collect.Iterables.getOnlyElement;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Type;
import java.util.List;
import java.util.Map;

/** Implementation for the {@code java_toolchain} rule. */
public class JavaToolchain implements RuleConfiguredTargetFactory {

  private final JavaSemantics semantics;

  protected JavaToolchain(JavaSemantics semantics) {
    this.semantics = semantics;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    ImmutableList<String> javacopts = getJavacOpts(ruleContext);
    BootClassPathInfo bootclasspath = getBootClassPathInfo(ruleContext);
    boolean javacSupportsWorkers =
        ruleContext.attributes().get("javac_supports_workers", Type.BOOLEAN);
    boolean javacSupportsMultiplexWorkers =
        ruleContext.attributes().get("javac_supports_multiplex_workers", Type.BOOLEAN);
    FilesToRunProvider javabuilder = ruleContext.getExecutablePrerequisite("javabuilder");
    FilesToRunProvider headerCompiler = ruleContext.getExecutablePrerequisite("header_compiler");
    FilesToRunProvider headerCompilerDirect =
        ruleContext.getExecutablePrerequisite("header_compiler_direct");
    ImmutableSet<String> headerCompilerBuiltinProcessors =
        ImmutableSet.copyOf(
            ruleContext.attributes().get("header_compiler_builtin_processors", Type.STRING_LIST));
    ImmutableSet<String> reducedClasspathIncompatibleProcessors =
        ImmutableSet.copyOf(
            ruleContext
                .attributes()
                .get("reduced_classpath_incompatible_processors", Type.STRING_LIST));
    boolean forciblyDisableHeaderCompilation =
        ruleContext.attributes().get("forcibly_disable_header_compilation", Type.BOOLEAN);
    Artifact singleJar = ruleContext.getPrerequisiteArtifact("singlejar");
    Artifact oneVersion = ruleContext.getPrerequisiteArtifact("oneversion");
    Artifact oneVersionAllowlist = ruleContext.getPrerequisiteArtifact("oneversion_whitelist");
    Artifact genClass = ruleContext.getPrerequisiteArtifact("genclass");
    Artifact resourceJarBuilder = ruleContext.getPrerequisiteArtifact("resourcejar");
    Artifact timezoneData = ruleContext.getPrerequisiteArtifact("timezone_data");
    FilesToRunProvider ijar = ruleContext.getExecutablePrerequisite("ijar");
    FilesToRunProvider proguardAllowlister =
        ruleContext.getExecutablePrerequisite("proguard_allowlister");
    ImmutableListMultimap<String, String> compatibleJavacOptions =
        getCompatibleJavacOptions(ruleContext);

    NestedSet<Artifact> tools = PrerequisiteArtifacts.nestedSet(ruleContext, "tools");

    ImmutableList<String> jvmOpts = getJvmOpts(ruleContext, "jvm_opts");
    ImmutableList<String> javabuilderJvmOpts =
        ImmutableList.<String>builder()
            .addAll(jvmOpts)
            .addAll(getJvmOpts(ruleContext, "javabuilder_jvm_opts"))
            .build();
    ImmutableList<String> turbineJvmOpts =
        ImmutableList.<String>builder()
            .addAll(jvmOpts)
            .addAll(getJvmOpts(ruleContext, "turbine_jvm_opts"))
            .build();

    ImmutableList<JavaPackageConfigurationProvider> packageConfiguration =
        ImmutableList.copyOf(
            ruleContext.getPrerequisites(
                "package_configuration",
                JavaPackageConfigurationProvider.class));

    FilesToRunProvider jacocoRunner = ruleContext.getExecutablePrerequisite("jacocorunner");

    JavaRuntimeInfo javaRuntime =
        (JavaRuntimeInfo) ruleContext.getPrerequisite("java_runtime").get(ToolchainInfo.PROVIDER);

    JavaToolchainProvider provider =
        JavaToolchainProvider.create(
            ruleContext.getLabel(),
            javacopts,
            jvmOpts,
            javabuilderJvmOpts,
            turbineJvmOpts,
            javacSupportsWorkers,
            javacSupportsMultiplexWorkers,
            bootclasspath,
            tools,
            javabuilder,
            headerCompiler,
            headerCompilerDirect,
            headerCompilerBuiltinProcessors,
            reducedClasspathIncompatibleProcessors,
            forciblyDisableHeaderCompilation,
            singleJar,
            oneVersion,
            oneVersionAllowlist,
            genClass,
            resourceJarBuilder,
            timezoneData,
            ijar,
            compatibleJavacOptions,
            packageConfiguration,
            jacocoRunner,
            proguardAllowlister,
            semantics,
            javaRuntime);
    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext)
            .addStarlarkTransitiveInfo(JavaToolchainProvider.LEGACY_NAME, provider)
            .addNativeDeclaredProvider(provider)
            .addProvider(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
            .setFilesToBuild(new NestedSetBuilder<Artifact>(Order.STABLE_ORDER).build());

    return builder.build();
  }

  private ImmutableList<String> getJavacOpts(RuleContext ruleContext) {
    ImmutableList.Builder<String> javacopts = ImmutableList.builder();
    String source = ruleContext.attributes().get("source_version", Type.STRING);
    if (!isNullOrEmpty(source)) {
      javacopts.add("-source").add(source);
    }
    String target = ruleContext.attributes().get("target_version", Type.STRING);
    if (!isNullOrEmpty(target)) {
      javacopts.add("-target").add(target);
    }
    List<String> xlint = ruleContext.attributes().get("xlint", Type.STRING_LIST);
    if (!xlint.isEmpty()) {
      javacopts.add("-Xlint:" + Joiner.on(",").join(xlint));
    }
    javacopts.addAll(ruleContext.getExpander().withDataLocations().tokenized("misc"));
    javacopts.addAll(ruleContext.getExpander().withDataLocations().tokenized("javacopts"));
    return javacopts.build();
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
      RuleContext ruleContext,
      String attribute) {
    return ruleContext.getExpander().withExecLocations(ImmutableMap.of()).list(attribute);
  }

  private static BootClassPathInfo getBootClassPathInfo(RuleContext ruleContext) {
    List<BootClassPathInfo> bootClassPathInfos =
        ruleContext.getPrerequisites("bootclasspath", BootClassPathInfo.PROVIDER);
    if (!bootClassPathInfos.isEmpty()) {
      if (bootClassPathInfos.size() != 1) {
        ruleContext.attributeError(
            "bootclasspath", "expected at most one entry with a BootClassPathInfo provider");
      }
      return getOnlyElement(bootClassPathInfos);
    }
    return BootClassPathInfo.create(PrerequisiteArtifacts.nestedSet(ruleContext, "bootclasspath"));
  }
}
