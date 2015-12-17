// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction.Builder;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaOptimizationMode;
import com.google.devtools.build.lib.rules.java.ProguardSpecProvider;

import javax.annotation.Nullable;

/**
 * Common code for proguarding Android binaries.
 */
public class ProguardHelper {

  static final String PROGUARD_SPECS = "proguard_specs";

  @Immutable
  static final class ProguardOutput {
    private final Artifact outputJar;
    @Nullable private final Artifact mapping;

    ProguardOutput(Artifact outputJar, @Nullable Artifact mapping) {
      this.outputJar = outputJar;
      this.mapping = mapping;
    }

    public Artifact getOutputJar() {
      return outputJar;
    }

    @Nullable
    public Artifact getMapping() {
      return mapping;
    }

  }

  private ProguardHelper() {}

  /**
   * Retrieves the full set of proguard specs that should be applied to this binary, including the
   * specs passed in, if Proguard should run on the given rule.  {@link #createProguardAction}
   * relies on this method returning an empty list if the given rule doesn't declare specs in
   * --java_optimization_mode=legacy.
   *
   * <p>If Proguard shouldn't be applied, or the legacy link mode is used and there are no
   * proguard_specs on this rule, an empty list will be returned, regardless of any given specs or
   * specs from dependencies.  {@link AndroidBinary#createAndroidBinary} relies on that behavior.
   */
  public static ImmutableList<Artifact> collectTransitiveProguardSpecs(RuleContext ruleContext,
      Artifact... specsToInclude) {
    JavaOptimizationMode optMode = getJavaOptimizationMode(ruleContext);
    if (optMode == JavaOptimizationMode.NOOP) {
      return ImmutableList.of();
    }

    ImmutableList<Artifact> proguardSpecs =
        ruleContext.attributes().has(PROGUARD_SPECS, BuildType.LABEL_LIST)
            ? ruleContext.getPrerequisiteArtifacts(PROGUARD_SPECS, Mode.TARGET).list()
            : ImmutableList.<Artifact>of();
    if (optMode == JavaOptimizationMode.LEGACY && proguardSpecs.isEmpty()) {
      return ImmutableList.of();
    }

    // TODO(bazel-team): In modes except LEGACY verify that proguard specs don't include -dont...
    // flags since those flags would override the desired optMode
    ImmutableSortedSet.Builder<Artifact> builder =
        ImmutableSortedSet.orderedBy(Artifact.EXEC_PATH_COMPARATOR)
            .addAll(proguardSpecs)
            .add(specsToInclude)
            .addAll(ruleContext
                .getPrerequisiteArtifacts(":extra_proguard_specs", Mode.TARGET)
                .list());
    for (ProguardSpecProvider dep :
        ruleContext.getPrerequisites("deps", Mode.TARGET, ProguardSpecProvider.class)) {
      builder.addAll(dep.getTransitiveProguardSpecs());
    }

    // Generate and include implicit Proguard spec for requested mode.
    if (!optMode.getImplicitProguardDirectives().isEmpty()) {
      Artifact implicitDirectives =
          getProguardConfigArtifact(ruleContext, optMode.name().toLowerCase());
      ruleContext.registerAction(
          new FileWriteAction(
              ruleContext.getActionOwner(),
              implicitDirectives,
              optMode.getImplicitProguardDirectives(),
              /*executable*/ false));
      builder.add(implicitDirectives);
    }

    return builder.build().asList();
  }

  /**
   * Creates an action to run Proguard over the given {@code programJar} with various other given
   * inputs to produce {@code proguardOutputJar}.  If requested explicitly, or implicitly with
   * --java_optimization_mode, the action also produces a mapping file (which shows what methods and
   * classes in the output Jar correspond to which methods and classes in the input).  The "pair"
   * returned by this method indicates whether a mapping is being produced.
   *
   * <p>See the Proguard manual for the meaning of the various artifacts in play.
   *
   * @param proguard Proguard executable to use
   * @param proguardSpecs Proguard specification files to pass to Proguard
   * @param proguardMapping optional mapping file for Proguard to apply
   * @param libraryJars any other Jar files that the {@code programJar} will run against
   * @param mappingRequested whether to ask Proguard to output a mapping file (a mapping will be
   *        produced anyway if --java_optimization_mode includes obfuscation)
   * @param filesBuilder all artifacts produced by this rule will be added to this builder
   */
  public static ProguardOutput createProguardAction(RuleContext ruleContext,
      FilesToRunProvider proguard,
      Artifact programJar,
      ImmutableList<Artifact> proguardSpecs,
      @Nullable Artifact proguardMapping,
      NestedSet<Artifact> libraryJars,
      Artifact proguardOutputJar,
      boolean mappingRequested,
      NestedSetBuilder<Artifact> filesBuilder) throws InterruptedException {
    JavaOptimizationMode optMode = getJavaOptimizationMode(ruleContext);
    Preconditions.checkArgument(optMode != JavaOptimizationMode.NOOP);
    Preconditions.checkArgument(optMode != JavaOptimizationMode.LEGACY || !proguardSpecs.isEmpty());

    Builder builder = new SpawnAction.Builder()
        .addInput(programJar)
        .addInputs(libraryJars)
        .addInputs(proguardSpecs)
        .addOutput(proguardOutputJar)
        .setExecutable(proguard)
        .setProgressMessage("Trimming binary with Proguard")
        .setMnemonic("Proguard")
        .addArgument("-injars")
        .addArgument(programJar.getExecPathString());

    for (Artifact libraryJar : libraryJars) {
      builder.addArgument("-libraryjars")
          .addArgument(libraryJar.getExecPathString());
    }

    filesBuilder.add(proguardOutputJar);

    if (proguardMapping != null) {
      builder.addInput(proguardMapping)
          .addArgument("-applymapping")
          .addArgument(proguardMapping.getExecPathString());
    }

    builder.addArgument("-outjars")
        .addArgument(proguardOutputJar.getExecPathString());

    for (Artifact proguardSpec : proguardSpecs) {
      builder.addArgument("@" + proguardSpec.getExecPathString());
    }

    Artifact proguardOutputMap = null;
    if (mappingRequested || optMode.alwaysGenerateOutputMapping()) {
      // TODO(bazel-team): Verify that proguard spec files don't contain -printmapping directions
      // which this -printmapping command line flag will override.
      proguardOutputMap = ruleContext.getImplicitOutputArtifact(
          AndroidRuleClasses.ANDROID_BINARY_PROGUARD_MAP);

      builder.addOutput(proguardOutputMap)
          .addArgument("-printmapping")
          .addArgument(proguardOutputMap.getExecPathString());
      filesBuilder.add(proguardOutputMap);
    }

    ruleContext.registerAction(builder.build(ruleContext));
    return new ProguardOutput(proguardOutputJar, proguardOutputMap);
  }

  /**
   * Returns an intermediate artifact used to run Proguard.
   */
  public static Artifact getProguardConfigArtifact(RuleContext ruleContext, String prefix) {
    // TODO(bazel-team): Remove the redundant inclusion of the rule name, as getUniqueDirectory
    // includes the rulename as well.
    return Preconditions.checkNotNull(ruleContext.getUniqueDirectoryArtifact(
        "proguard",
        Joiner.on("_").join(prefix, ruleContext.getLabel().getName(), "proguard.cfg"),
        ruleContext.getBinOrGenfilesDirectory()));
  }

  private static JavaOptimizationMode getJavaOptimizationMode(RuleContext ruleContext) {
    return ruleContext.getConfiguration().getFragment(JavaConfiguration.class)
        .getJavaOptimizationMode();
  }
}
