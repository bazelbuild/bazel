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
package com.google.devtools.build.lib.rules.java;

import static com.google.common.base.Preconditions.checkNotNull;

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
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaOptimizationMode;

import javax.annotation.Nullable;

/**
 * Common code for proguarding Java binaries.
 */
public abstract class ProguardHelper {

  /**
   * Attribute for attaching proguard specs explicitly to a rule, if such an attribute is desired.
   */
  public static final String PROGUARD_SPECS = "proguard_specs";

  /**
   * Pair summarizing Proguard's output: a Jar file and an optional obfuscation mapping file.
   */
  @Immutable
  public static final class ProguardOutput {
    private final Artifact outputJar;
    @Nullable private final Artifact mapping;

    public ProguardOutput(Artifact outputJar, @Nullable Artifact mapping) {
      this.outputJar = checkNotNull(outputJar);
      this.mapping = mapping;
    }

    public Artifact getOutputJar() {
      return outputJar;
    }

    @Nullable
    public Artifact getMapping() {
      return mapping;
    }

    /** Adds the output artifacts to the given set builder. */
    public void addAllToSet(NestedSetBuilder<Artifact> filesBuilder) {
      filesBuilder.add(outputJar);
      if (mapping != null) {
        filesBuilder.add(mapping);
      }
    }
  }

  protected ProguardHelper() {}

  /**
   * Creates an action to run Proguard to <i>output</i> the given {@code deployJar} artifact
   * if --java_optimization_mode calls for it from an assumed input artifact
   * {@link JavaSemantics#JAVA_BINARY_MERGED_JAR}.  Returns the artifacts that Proguard will
   * generate or {@code null} if Proguard isn't used.
   *
   * <p>If this method returns artifacts then {@link DeployArchiveBuilder} needs to write the
   * assumed input artifact (instead of the conventional deploy.jar, which now Proguard writes).
   * Do not use this method for binary rules that themselves declare {@link #PROGUARD_SPECS}
   * attributes, which includes of 1/2016 {@code android_binary} and {@code android_test}.
   */
  @Nullable
  public ProguardOutput applyProguardIfRequested(
      RuleContext ruleContext,
      Artifact deployJar,
      ImmutableList<Artifact> bootclasspath,
      String mainClassName) throws InterruptedException {
    JavaOptimizationMode optMode = getJavaOptimizationMode(ruleContext);
    if (optMode == JavaOptimizationMode.NOOP || optMode == JavaOptimizationMode.LEGACY) {
      // For simplicity do nothing in LEGACY mode
      return null;
    }

    Preconditions.checkArgument(!bootclasspath.isEmpty(), "Bootclasspath should not be empty");
    FilesToRunProvider proguard = findProguard(ruleContext);
    if (proguard == null) {
      ruleContext.ruleError("--proguard_top required for --java_optimization_mode=" + optMode);
      return null;
    }

    ImmutableList<Artifact> proguardSpecs =
        collectProguardSpecs(ruleContext, bootclasspath, mainClassName);
    Artifact singleJar =
        ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_MERGED_JAR);
    return createProguardAction(ruleContext, proguard, singleJar, proguardSpecs, (Artifact) null,
        bootclasspath, deployJar, /* mappingRequested */ false, /* optimizationPases */ null);
  }

  private ImmutableList<Artifact> collectProguardSpecs(
      RuleContext ruleContext, ImmutableList<Artifact> bootclasspath, String mainClassName) {
    return ProguardHelper.collectTransitiveProguardSpecs(
        ruleContext, collectProguardSpecsForRule(ruleContext, bootclasspath, mainClassName));
  }

  /**
   * Returns the Proguard binary to invoke when using {@link #applyProguardIfRequested}.  Returning
   * {@code null} from this method will generate an error in that method.
   *
   * @return Proguard binary or {@code null} if none is available
   */
  @Nullable
  protected abstract FilesToRunProvider findProguard(RuleContext ruleContext);

  /**
   * Returns rule-specific proguard specs not captured by {@link #PROGUARD_SPECS} attributes when
   * using {@link #applyProguardIfRequested}.  Typically these are generated artifacts such as specs
   * generated for android resources. This method is only called if Proguard will definitely used,
   * so it's ok to generate files here.
   */
  protected abstract ImmutableList<Artifact> collectProguardSpecsForRule(
      RuleContext ruleContext, ImmutableList<Artifact> bootclasspath, String mainClassName);

  /**
   * Retrieves the full set of proguard specs that should be applied to this binary, including the
   * specs passed in, if Proguard should run on the given rule.  {@link #createProguardAction}
   * relies on this method returning an empty list if the given rule doesn't declare specs in
   * --java_optimization_mode=legacy.
   *
   * <p>If Proguard shouldn't be applied, or the legacy link mode is used and there are no
   * proguard_specs on this rule, an empty list will be returned, regardless of any given specs or
   * specs from dependencies.
   * {@link com.google.devtools.build.lib.rules.android.AndroidBinary#createAndroidBinary} relies on
   * that behavior.
   */
  public static ImmutableList<Artifact> collectTransitiveProguardSpecs(RuleContext ruleContext,
      Iterable<Artifact> specsToInclude) {
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
            .addAll(specsToInclude)
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
   * Creates a proguard spec that tells proguard to keep the binary's entry point, ie., the
   * {@code main()} method to be invoked.
   */
  protected static Artifact generateSpecForJavaBinary(
      RuleContext ruleContext, String mainClassName) {
    Artifact result = ProguardHelper.getProguardConfigArtifact(ruleContext, "jvm");
    ruleContext.registerAction(
        new FileWriteAction(
            ruleContext.getActionOwner(),
            result,
            String.format(
                "-keep class %s {%n"
                    + "  public static void main(java.lang.String[]);%n"
                    + "}",
                mainClassName),
            /*executable*/ false));
    return result;
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
   * @param optimizationPasses if not null specifies to break proguard up into multiple passes with
   *        the given number of optimization passes.
   */
  public static ProguardOutput createProguardAction(RuleContext ruleContext,
      FilesToRunProvider proguard,
      Artifact programJar,
      ImmutableList<Artifact> proguardSpecs,
      @Nullable Artifact proguardMapping,
      Iterable<Artifact> libraryJars,
      Artifact proguardOutputJar,
      boolean mappingRequested,
      @Nullable Integer optimizationPasses) throws InterruptedException {
    JavaOptimizationMode optMode = getJavaOptimizationMode(ruleContext);
    Preconditions.checkArgument(optMode != JavaOptimizationMode.NOOP);
    Preconditions.checkArgument(optMode != JavaOptimizationMode.LEGACY || !proguardSpecs.isEmpty());

    Artifact proguardOutputMap;
    if (mappingRequested || optMode.alwaysGenerateOutputMapping()) {
      // TODO(bazel-team): Verify that proguard spec files don't contain -printmapping directions
      // which this -printmapping command line flag will override.
      proguardOutputMap =
          ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_PROGUARD_MAP);
    } else {
      proguardOutputMap = null;
    }

    if (optimizationPasses == null) {
      // Run proguard as a single step.
      Builder builder = makeBuilder(
          proguard,
          programJar,
          proguardSpecs,
          proguardMapping,
          libraryJars,
          proguardOutputJar,
          proguardOutputMap)
          .setProgressMessage("Trimming binary with Proguard")
          .addOutput(proguardOutputJar);

      if (proguardOutputMap != null) {
        builder.addOutput(proguardOutputMap);
      }

      ruleContext.registerAction(builder.build(ruleContext));
    } else {
      // Optimization passes have been specified, so run proguard in multiple phases.

      Artifact lastStageOutput = getProguardTempArtifact(
          ruleContext, optMode.name().toLowerCase(), "proguard_preoptimization.jar");
      ruleContext.registerAction(
          makeBuilder(
              proguard,
              programJar,
              proguardSpecs,
              proguardMapping,
              libraryJars,
              proguardOutputJar,
              proguardOutputMap)
              .setProgressMessage("Trimming binary with Proguard: Verification/Shrinking Pass")
              .addArgument("-runtype INITIAL")
              .addOutput(lastStageOutput)
              .addArgument("-nextstageoutput")
              .addArgument(lastStageOutput.getExecPathString())
              .build(ruleContext));
      for (int i = 0; i < optimizationPasses; i++) {
        Artifact optimizationOutput = getProguardTempArtifact(
            ruleContext, optMode.name().toLowerCase(), "proguard_optimization_" + (i + 1) + ".jar");
        ruleContext.registerAction(
            makeBuilder(
                proguard,
                programJar,
                proguardSpecs,
                proguardMapping,
                libraryJars,
                proguardOutputJar,
                proguardOutputMap)
                .setProgressMessage("Trimming binary with Proguard: Optimization Pass " + (i + 1))
                .addArgument("-runtype OPTIMIZATION")
                .addInput(lastStageOutput)
                .addArgument("-laststageoutput")
                .addArgument(lastStageOutput.getExecPathString())
                .addOutput(optimizationOutput)
                .addArgument("-nextstageoutput")
                .addArgument(optimizationOutput.getExecPathString())
                .build(ruleContext));
        lastStageOutput = optimizationOutput;
      }

      Builder builder = makeBuilder(
          proguard,
          programJar,
          proguardSpecs,
          proguardMapping,
          libraryJars,
          proguardOutputJar,
          proguardOutputMap)
          .setProgressMessage("Trimming binary with Proguard: Obfuscation and Final Ouput Pass")
          .addArgument("-runtype FINAL")
          .addInput(lastStageOutput)
          .addArgument("-laststageoutput")
          .addArgument(lastStageOutput.getExecPathString())
          .addOutput(proguardOutputJar);

      if (proguardOutputMap != null) {
        builder.addOutput(proguardOutputMap);
      }

      ruleContext.registerAction(builder.build(ruleContext));
    }

    return new ProguardOutput(proguardOutputJar, proguardOutputMap);
  }

  private static Builder makeBuilder(
      FilesToRunProvider proguard,
      Artifact programJar,
      ImmutableList<Artifact> proguardSpecs,
      @Nullable Artifact proguardMapping,
      Iterable<Artifact> libraryJars,
      Artifact proguardOutputJar,
      @Nullable Artifact proguardOutputMap) {
    Builder builder = new SpawnAction.Builder()
        .addInput(programJar)
        .addInputs(libraryJars)
        .addInputs(proguardSpecs)
        .setExecutable(proguard)
        .setMnemonic("Proguard")
        .addArgument("-forceprocessing")
        .addArgument("-injars")
        .addArgument(programJar.getExecPathString())
        // This is handled by the build system there is no need for proguard to check if things are
        // up to date.
        .addArgument("-outjars")
        .addArgument(proguardOutputJar.getExecPathString());

    for (Artifact libraryJar : libraryJars) {
      builder.addArgument("-libraryjars")
          .addArgument(libraryJar.getExecPathString());
    }

    if (proguardMapping != null) {
      builder.addInput(proguardMapping)
          .addArgument("-applymapping")
          .addArgument(proguardMapping.getExecPathString());
    }

    for (Artifact proguardSpec : proguardSpecs) {
      builder.addArgument("@" + proguardSpec.getExecPathString());
    }

    if (proguardOutputMap != null) {
      builder
          .addArgument("-printmapping")
          .addArgument(proguardOutputMap.getExecPathString());
    }

    return builder;
  }

  /**
   * Returns an intermediate artifact used to run Proguard.
   */
  public static Artifact getProguardTempArtifact(
      RuleContext ruleContext, String prefix, String name) {
    // TODO(bazel-team): Remove the redundant inclusion of the rule name, as getUniqueDirectory
    // includes the rulename as well.
    return Preconditions.checkNotNull(ruleContext.getUniqueDirectoryArtifact(
        "proguard",
        Joiner.on("_").join(prefix, ruleContext.getLabel().getName(), name),
        ruleContext.getBinOrGenfilesDirectory()));
  }

  public static Artifact getProguardConfigArtifact(RuleContext ruleContext, String prefix) {
    return getProguardTempArtifact(ruleContext, prefix, "proguard.cfg");
  }

  /**
   * Returns {@link JavaConfiguration#getJavaOptimizationMode()}.
   */
  public static JavaOptimizationMode getJavaOptimizationMode(RuleContext ruleContext) {
    return ruleContext.getConfiguration().getFragment(JavaConfiguration.class)
        .getJavaOptimizationMode();
  }
}
