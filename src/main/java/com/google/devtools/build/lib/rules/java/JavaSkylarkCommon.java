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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.skylark.SkylarkActionFactory;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaCommonApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import java.util.List;
import javax.annotation.Nullable;

/** A module that contains Skylark utilities for Java support. */
public class JavaSkylarkCommon
    implements JavaCommonApi<
        Artifact,
        JavaInfo,
        JavaToolchainProvider,
        JavaRuntimeInfo,
        SkylarkRuleContext,
        SkylarkActionFactory> {
  private final JavaSemantics javaSemantics;

  @Override
  public JavaInfo create(
      @Nullable Object actionsUnchecked,
      Object compileTimeJars,
      Object runtimeJars,
      Boolean useIjar,
      @Nullable Object javaToolchain,
      Object transitiveCompileTimeJars,
      Object transitiveRuntimeJars,
      Object sourceJars,
      Location location,
      Environment environment)
      throws EvalException {
    if (environment.getSemantics().incompatibleDisallowLegacyJavaInfo()) {
      checkCallPathInWhitelistedPackages(
          environment.getSemantics(),
          location,
          environment.getCallerLabel().getPackageFragment().toString());
    }
    return JavaInfoBuildHelper.getInstance()
        .create(
            actionsUnchecked,
            asArtifactNestedSet(compileTimeJars),
            asArtifactNestedSet(runtimeJars),
            useIjar,
            javaToolchain == Runtime.NONE ? null : (JavaToolchainProvider) javaToolchain,
            asArtifactNestedSet(transitiveCompileTimeJars),
            asArtifactNestedSet(transitiveRuntimeJars),
            asArtifactNestedSet(sourceJars),
            environment.getSemantics(),
            location);
  }

  public JavaSkylarkCommon(JavaSemantics javaSemantics) {
    this.javaSemantics = javaSemantics;
  }

  @Override
  public Provider getJavaProvider() {
    return JavaInfo.PROVIDER;
  }

  @Override
  public JavaInfo createJavaCompileAction(
      SkylarkRuleContext skylarkRuleContext,
      SkylarkList<Artifact> sourceJars,
      SkylarkList<Artifact> sourceFiles,
      Artifact outputJar,
      Object outputSourceJar,
      SkylarkList<String> javacOpts,
      SkylarkList<JavaInfo> deps,
      SkylarkList<JavaInfo> exports,
      SkylarkList<JavaInfo> plugins,
      SkylarkList<JavaInfo> exportedPlugins,
      String strictDepsMode,
      JavaToolchainProvider javaToolchain,
      JavaRuntimeInfo hostJavabase,
      SkylarkList<Artifact> sourcepathEntries,
      SkylarkList<Artifact> resources,
      Boolean neverlink,
      Location location,
      Environment environment)
      throws EvalException, InterruptedException {

    return JavaInfoBuildHelper.getInstance()
        .createJavaCompileAction(
            skylarkRuleContext,
            sourceJars,
            sourceFiles,
            outputJar,
            outputSourceJar == Runtime.NONE ? null : (Artifact) outputSourceJar,
            javacOpts,
            deps,
            exports,
            plugins,
            exportedPlugins,
            strictDepsMode,
            javaToolchain,
            hostJavabase,
            sourcepathEntries,
            resources,
            neverlink,
            javaSemantics,
            location,
            environment);
  }

  @Override
  public Artifact runIjar(
      SkylarkActionFactory actions,
      Artifact jar,
      Object targetLabel,
      JavaToolchainProvider javaToolchain,
      Location location,
      StarlarkSemantics semantics)
      throws EvalException {
    return JavaInfoBuildHelper.getInstance()
        .buildIjar(
            actions,
            jar,
            targetLabel != Runtime.NONE ? (Label) targetLabel : null,
            javaToolchain,
            location);
  }

  @Override
  public Artifact stampJar(
      SkylarkActionFactory actions,
      Artifact jar,
      Label targetLabel,
      JavaToolchainProvider javaToolchain,
      Location location,
      StarlarkSemantics semantics)
      throws EvalException {
    return JavaInfoBuildHelper.getInstance()
        .stampJar(actions, jar, targetLabel, javaToolchain, location);
  }

  @Override
  public Artifact packSources(
      SkylarkActionFactory actions,
      Artifact outputJar,
      SkylarkList<Artifact> sourceFiles,
      SkylarkList<Artifact> sourceJars,
      JavaToolchainProvider javaToolchain,
      JavaRuntimeInfo hostJavabase,
      Location location,
      StarlarkSemantics semantics)
      throws EvalException {
    return JavaInfoBuildHelper.getInstance()
        .packSourceFiles(
            actions,
            outputJar,
            /* outputSourceJar= */ null,
            sourceFiles,
            sourceJars,
            javaToolchain,
            hostJavabase,
            location);
  }

  @Override
  // TODO(b/78512644): migrate callers to passing explicit javacopts or using custom toolchains, and
  // delete
  public ImmutableList<String> getDefaultJavacOpts(
      JavaToolchainProvider javaToolchain, Location location) throws EvalException {
    // We don't have a rule context if the default_javac_opts.java_toolchain parameter is set
    return ((JavaToolchainProvider) javaToolchain).getJavacOptions(/* ruleContext= */ null);
  }

  @Override
  public JavaInfo mergeJavaProviders(SkylarkList<JavaInfo> providers) {
    return JavaInfo.merge(providers);
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

  /**
   * Takes an Object that is either a SkylarkNestedSet or a SkylarkList of Artifacts and returns it
   * as a NestedSet.
   */
  private NestedSet<Artifact> asArtifactNestedSet(Object o) throws EvalException {
    return o instanceof SkylarkNestedSet
        ? ((SkylarkNestedSet) o).getSet(Artifact.class)
        : NestedSetBuilder.<Artifact>naiveLinkOrder()
            .addAll(((SkylarkList<?>) o).getContents(Artifact.class, /*description=*/ null))
            .build();
  }

  /**
   * Throws an {@link EvalException} if the given {@code callPath} is not listed under the {@code
   * --experimental_java_common_create_provider_enabled_packages} flag.
   */
  private static void checkCallPathInWhitelistedPackages(
      StarlarkSemantics semantics, Location location, String callPath) throws EvalException {
    List<String> whitelistedPackagesList =
        semantics.experimentalJavaCommonCreateProviderEnabledPackages();
    if (whitelistedPackagesList.stream().noneMatch(path -> callPath.startsWith(path))) {
      throw new EvalException(
          location,
          "java_common.create_provider is deprecated and cannot be used when "
              + "--incompatible_disallow_legacy_javainfo is set. "
              + "Please migrate to the JavaInfo constructor.");
    }
  }

  @Override
  public boolean isJavaToolchainResolutionEnabled(SkylarkRuleContext ruleContext)
      throws EvalException {
    return ruleContext
        .getConfiguration()
        .getOptions()
        .get(PlatformOptions.class)
        .useToolchainResolutionForJavaRules;
  }
}
