// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.packages.BazelModuleContext;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaCommonApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaToolchainStarlarkApiProviderApi;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;

/** A module that contains Starlark utilities for Java support. */
public class JavaStarlarkCommon
    implements JavaCommonApi<
        Artifact,
        JavaInfo,
        JavaToolchainProvider,
        ConstraintValueInfo,
        StarlarkRuleContext,
        StarlarkActionFactory> {
  private final JavaSemantics javaSemantics;

  private static final ImmutableSet<String> PRIVATE_STARLARKIFICATION_ALLOWLIST =
      ImmutableSet.of(
          "@_builtins//bazel_internal/test_rules/java:java_library.bzl",
          "//tools/build_defs/java:java_library.bzl");

  public JavaStarlarkCommon(JavaSemantics javaSemantics) {
    this.javaSemantics = javaSemantics;
  }

  @Override
  public Provider getJavaProvider() {
    return JavaInfo.PROVIDER;
  }

  @Override
  public JavaInfo createJavaCompileAction(
      StarlarkRuleContext starlarkRuleContext,
      Sequence<?> sourceJars, // <Artifact> expected
      Sequence<?> sourceFiles, // <Artifact> expected
      Artifact outputJar,
      Object outputSourceJar,
      Sequence<?> javacOpts, // <String> expected
      Sequence<?> deps, // <JavaInfo> expected
      Sequence<?> runtimeDeps, // <JavaInfo> expected
      Sequence<?> experimentalLocalCompileTimeDeps, // <JavaInfo> expected
      Sequence<?> exports, // <JavaInfo> expected
      Sequence<?> plugins, // <JavaInfo> expected
      Sequence<?> exportedPlugins, // <JavaInfo> expected
      Sequence<?> annotationProcessorAdditionalInputs, // <Artifact> expected
      Sequence<?> annotationProcessorAdditionalOutputs, // <Artifact> expected
      String strictDepsMode,
      JavaToolchainProvider javaToolchain,
      Object hostJavabase,
      Sequence<?> sourcepathEntries, // <Artifact> expected
      Sequence<?> resources, // <Artifact> expected
      Boolean neverlink,
      StarlarkThread thread)
      throws EvalException, InterruptedException {

    Sequence<JavaInfo> exportsJavaInfo;
    Sequence<Label> exportsLabels;
    if (exports.isEmpty() || exports.get(0) instanceof JavaInfo) {
      exportsLabels = StarlarkList.empty();
      exportsJavaInfo = Sequence.cast(exports, JavaInfo.class, "exports");
    } else {
      Label label =
          ((BazelModuleContext) Module.ofInnermostEnclosingStarlarkFunction(thread).getClientData())
              .label();
      if (!PRIVATE_STARLARKIFICATION_ALLOWLIST.contains(label.toString())) {
        throw Starlark.errorf("Rule in '%s' cannot use private API", label.getPackageName());
      }
      Sequence<TransitiveInfoCollection> e =
          Sequence.cast(exports, TransitiveInfoCollection.class, "exports");
      exportsLabels =
          StarlarkList.immutableCopyOf(
              e.stream().map(TransitiveInfoCollection::getLabel).collect(toImmutableList()));
      exportsJavaInfo =
          StarlarkList.immutableCopyOf(
              e.stream().map(JavaInfo::getJavaInfo).collect(toImmutableList()));
    }

    return JavaInfoBuildHelper.getInstance()
        .createJavaCompileAction(
            starlarkRuleContext,
            Sequence.cast(sourceJars, Artifact.class, "source_jars"),
            Sequence.cast(sourceFiles, Artifact.class, "source_files"),
            outputJar,
            outputSourceJar == Starlark.NONE ? null : (Artifact) outputSourceJar,
            Sequence.cast(javacOpts, String.class, "javac_opts"),
            Sequence.cast(deps, JavaInfo.class, "deps"),
            Sequence.cast(runtimeDeps, JavaInfo.class, "runtime_deps"),
            Sequence.cast(
                experimentalLocalCompileTimeDeps,
                JavaInfo.class,
                "experimental_local_compile_time_deps"),
            exportsJavaInfo,
            exportsLabels,
            Sequence.cast(plugins, JavaInfo.class, "plugins"),
            Sequence.cast(exportedPlugins, JavaInfo.class, "exported_plugins"),
            Sequence.cast(
                annotationProcessorAdditionalInputs,
                Artifact.class,
                "annotation_processor_additional_inputs"),
            Sequence.cast(
                annotationProcessorAdditionalOutputs,
                Artifact.class,
                "annotation_processor_additional_outputs"),
            strictDepsMode,
            javaToolchain,
            ImmutableList.copyOf(Sequence.cast(sourcepathEntries, Artifact.class, "sourcepath")),
            Sequence.cast(resources, Artifact.class, "resources"),
            neverlink,
            javaSemantics,
            thread);
  }

  @Override
  public Artifact runIjar(
      StarlarkActionFactory actions,
      Artifact jar,
      Object targetLabel,
      JavaToolchainProvider javaToolchain)
      throws EvalException {
    return JavaInfoBuildHelper.getInstance()
        .buildIjar(
            actions, jar, targetLabel != Starlark.NONE ? (Label) targetLabel : null, javaToolchain);
  }

  @Override
  public Artifact stampJar(
      StarlarkActionFactory actions,
      Artifact jar,
      Label targetLabel,
      JavaToolchainProvider javaToolchain)
      throws EvalException {
    return JavaInfoBuildHelper.getInstance().stampJar(actions, jar, targetLabel, javaToolchain);
  }

  @Override
  public Artifact packSources(
      StarlarkActionFactory actions,
      Object outputJar,
      Object outputSourceJar,
      Sequence<?> sourceFiles, // <Artifact> expected.
      Sequence<?> sourceJars, // <Artifact> expected.
      JavaToolchainProvider javaToolchain,
      Object hostJavabase)
      throws EvalException {
    return JavaInfoBuildHelper.getInstance()
        .packSourceFiles(
            actions,
            outputJar instanceof Artifact ? (Artifact) outputJar : null,
            outputSourceJar instanceof Artifact ? (Artifact) outputSourceJar : null,
            Sequence.cast(sourceFiles, Artifact.class, "sources"),
            Sequence.cast(sourceJars, Artifact.class, "source_jars"),
            javaToolchain);
  }

  @Override
  // TODO(b/78512644): migrate callers to passing explicit javacopts or using custom toolchains, and
  // delete
  public ImmutableList<String> getDefaultJavacOpts(JavaToolchainProvider javaToolchain)
      throws EvalException {
    // We don't have a rule context if the default_javac_opts.java_toolchain parameter is set
    return ((JavaToolchainProvider) javaToolchain).getJavacOptions(/* ruleContext= */ null);
  }

  @Override
  public JavaInfo mergeJavaProviders(Sequence<?> providers /* <JavaInfo> expected. */)
      throws EvalException {
    return JavaInfo.merge(Sequence.cast(providers, JavaInfo.class, "providers"));
  }

  // TODO(b/65113771): Remove this method because it's incorrect.
  @Override
  public JavaInfo makeNonStrict(JavaInfo javaInfo) {
    return JavaInfo.Builder.copyOf(javaInfo)
        // Overwrites the old provider.
        .addProvider(
            JavaCompilationArgsProvider.class,
            JavaCompilationArgsProvider.makeNonStrict(
                javaInfo.getProvider(JavaCompilationArgsProvider.class)))
        .build();
  }

  @Override
  public Provider getJavaToolchainProvider() {
    return ToolchainInfo.PROVIDER;
  }

  @Override
  public Provider getJavaRuntimeProvider() {
    return ToolchainInfo.PROVIDER;
  }

  @Override
  public boolean isJavaToolchainResolutionEnabled(StarlarkRuleContext ruleContext)
      throws EvalException {
    return ruleContext
        .getConfiguration()
        .getOptions()
        .get(PlatformOptions.class)
        .useToolchainResolutionForJavaRules;
  }

  @Override
  public ProviderApi getMessageBundleInfo() {
    // No implementation in Bazel. This method not callable in Starlark except through
    // (discouraged) use of --experimental_google_legacy_api.
    return null;
  }

  @Override
  public JavaInfo addConstraints(JavaInfo javaInfo, Sequence<?> constraints) throws EvalException {
    // No implementation in Bazel. This method not callable in Starlark except through
    // (discouraged) use of --experimental_google_legacy_api.
    return null;
  }

  @Override
  public Sequence<String> getConstraints(JavaInfo javaInfo) {
    // No implementation in Bazel. This method not callable in Starlark except through
    // (discouraged) use of --experimental_google_legacy_api.
    return StarlarkList.empty();
  }

  @Override
  public JavaInfo removeAnnotationProcessors(JavaInfo javaInfo) {
    // No implementation in Bazel. This method not callable in Starlark except through
    // (discouraged) use of --experimental_google_legacy_api.
    return null;
  }

  @Override
  public JavaInfo setAnnotationProcessing(
      JavaInfo javaInfo,
      boolean enabled,
      Sequence<?> processorClassnames,
      Object processorClasspath,
      Object classJar,
      Object sourceJar)
      throws EvalException {
    // No implementation in Bazel. This method not callable in Starlark except through
    // (discouraged) use of --experimental_google_legacy_api.
    return null;
  }

  @Override
  public Depset /*<Artifact>*/ getCompileTimeJavaDependencyArtifacts(JavaInfo javaInfo) {
    // No implementation in Bazel. This method not callable in Starlark except through
    // (discouraged) use of --experimental_google_legacy_api.
    return null;
  }

  @Override
  public JavaInfo addCompileTimeJavaDependencyArtifacts(
      JavaInfo javaInfo, Sequence<?> compileTimeJavaDependencyArtifacts) throws EvalException {
    // No implementation in Bazel. This method not callable in Starlark except through
    // (discouraged) use of --experimental_google_legacy_api.
    return null;
  }

  @Override
  public Label getJavaToolchainLabel(JavaToolchainStarlarkApiProviderApi toolchain)
      throws EvalException {
    // No implementation in Bazel. This method not callable in Starlark except through
    // (discouraged) use of --experimental_google_legacy_api.
    return null;
  }

  @Override
  public ProviderApi getBootClassPathInfo() {
    return BootClassPathInfo.PROVIDER;
  }
}
