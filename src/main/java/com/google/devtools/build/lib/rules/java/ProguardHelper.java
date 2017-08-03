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
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction.Builder;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaOptimizationMode;
import com.google.devtools.build.lib.syntax.Type;
import java.util.Map;
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
   * A class collecting Proguard output artifacts.
   */
  @Immutable
  public static final class ProguardOutput {
    private final Artifact outputJar;
    @Nullable private final Artifact mapping;
    @Nullable private final Artifact protoMapping;
    @Nullable private final Artifact seeds;
    @Nullable private final Artifact usage;
    @Nullable private final Artifact constantStringObfuscatedMapping;
    private final Artifact config;

    public ProguardOutput(Artifact outputJar,
                          @Nullable Artifact mapping,
                          @Nullable Artifact protoMapping,
                          @Nullable Artifact seeds,
                          @Nullable Artifact usage,
                          @Nullable Artifact constantStringObfuscatedMapping,
                          Artifact config) {
      this.outputJar = checkNotNull(outputJar);
      this.mapping = mapping;
      this.protoMapping = protoMapping;
      this.seeds = seeds;
      this.usage = usage;
      this.constantStringObfuscatedMapping = constantStringObfuscatedMapping;
      this.config = config;
    }

    public Artifact getOutputJar() {
      return outputJar;
    }

    @Nullable
    public Artifact getMapping() {
      return mapping;
    }

    @Nullable
    public Artifact getProtoMapping() {
      return protoMapping;
    }

    @Nullable
    public Artifact getConstantStringObfuscatedMapping() {
      return constantStringObfuscatedMapping;
    }

    @Nullable
    public Artifact getSeeds() {
      return seeds;
    }

    @Nullable
    public Artifact getUsage() {
      return usage;
    }

    public Artifact getConfig() {
      return config;
    }

    /** Adds the output artifacts to the given set builder. */
    public void addAllToSet(NestedSetBuilder<Artifact> filesBuilder) {
      addAllToSet(filesBuilder, null);
    }

    /** Adds the output artifacts to the given set builder. If the progaurd map was updated
     * then add the updated map instead of the original proguard output map */
    public void addAllToSet(NestedSetBuilder<Artifact> filesBuilder, Artifact finalProguardMap) {
      filesBuilder.add(outputJar);
      if (protoMapping != null) {
        filesBuilder.add(protoMapping);
      }
      if (constantStringObfuscatedMapping != null) {
        filesBuilder.add(constantStringObfuscatedMapping);
      }
      if (seeds != null) {
        filesBuilder.add(seeds);
      }
      if (usage != null) {
        filesBuilder.add(usage);
      }
      if (config != null) {
        filesBuilder.add(config);
      }
      if (finalProguardMap != null) {
        filesBuilder.add(finalProguardMap);
      } else if (mapping != null) {
        filesBuilder.add(mapping);
      }
    }
  }

  protected ProguardHelper() {}

  /**
   * Creates an action to run Proguard to <i>output</i> the given {@code deployJar} artifact if
   * --java_optimization_mode calls for it from an assumed input artifact {@link
   * JavaSemantics#JAVA_BINARY_MERGED_JAR}. Returns the artifacts that Proguard will generate or
   * {@code null} if Proguard isn't used.
   *
   * <p>If this method returns artifacts then {@link
   * com.google.devtools.build.lib.rules.java.DeployArchiveBuilder} needs to write the assumed input
   * artifact (instead of the conventional deploy.jar, which now Proguard writes). Do not use this
   * method for binary rules that themselves declare {@link #PROGUARD_SPECS} attributes, which as of
   * includes 1/2016 {@code android_binary} and {@code android_test}.
   */
  @Nullable
  public ProguardOutput applyProguardIfRequested(
      RuleContext ruleContext,
      Artifact deployJar,
      ImmutableList<Artifact> bootclasspath,
      String mainClassName,
      JavaSemantics semantics)
      throws InterruptedException {
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

    // TODO(bazel-team): Verify that proguard spec files don't contain -printmapping directions
    // which this -printmapping command line flag will override.
    Artifact proguardOutputMap = null;
    if (genProguardMapping(ruleContext.attributes()) || optMode.alwaysGenerateOutputMapping()) {
      proguardOutputMap =
          ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_PROGUARD_MAP);
    }
    return createOptimizationActions(
        ruleContext,
        proguard,
        singleJar,
        proguardSpecs,
        /* proguardSeeds */ (Artifact) null,
        /* proguardUsage */ (Artifact) null,
        /* proguardMapping */ (Artifact) null,
        bootclasspath,
        deployJar,
        semantics,
        /* optimizationPases */ 3,
        proguardOutputMap);
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
          FileWriteAction.create(
              ruleContext,
              implicitDirectives,
              optMode.getImplicitProguardDirectives(),
              /*makeExecutable=*/ false));
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
        FileWriteAction.create(
            ruleContext,
            result,
            String.format(
                "-keep class %s {%n  public static void main(java.lang.String[]);%n}",
                mainClassName),
            /*makeExecutable=*/ false));
    return result;
  }

  /**
   * @return true if proguard_generate_mapping is specified.
   */
  public static final boolean genProguardMapping(AttributeMap rule) {
      return rule.has("proguard_generate_mapping", Type.BOOLEAN)
          && rule.get("proguard_generate_mapping", Type.BOOLEAN);
  }

  public static final boolean genObfuscatedConstantStringMap(AttributeMap rule) {
    return rule.has("proguard_generate_obfuscated_constant_string_mapping", Type.BOOLEAN)
          && rule.get("proguard_generate_obfuscated_constant_string_mapping", Type.BOOLEAN);
  }

  public static ProguardOutput getProguardOutputs(
      Artifact outputJar,
      @Nullable Artifact proguardSeeds,
      @Nullable Artifact proguardUsage,
      RuleContext ruleContext,
      JavaSemantics semantics,
      @Nullable Artifact proguardOutputMap)
      throws InterruptedException {
    JavaOptimizationMode optMode = getJavaOptimizationMode(ruleContext);
    boolean mappingRequested = genProguardMapping(ruleContext.attributes());

    Artifact proguardOutputProtoMap = null;
    Artifact proguardConstantStringMap = null;

    if (mappingRequested || optMode.alwaysGenerateOutputMapping()) {
      // TODO(bazel-team): if rex is enabled, the proguard map will change and then will no
      // longer correspond to the proto map
      proguardOutputProtoMap = semantics.getProtoMapping(ruleContext);
    }

    if (genObfuscatedConstantStringMap(ruleContext.attributes())) {
      proguardConstantStringMap = semantics.getObfuscatedConstantStringMap(ruleContext);
    }

    Artifact proguardConfigOutput =
        ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_PROGUARD_CONFIG);

    return new ProguardOutput(
        outputJar,
        proguardOutputMap,
        proguardOutputProtoMap,
        proguardSeeds,
        proguardUsage,
        proguardConstantStringMap,
        proguardConfigOutput);
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
   * @param optimizationPasses if not null specifies to break proguard up into multiple passes with
   *        the given number of optimization passes.
   * @param proguardOutputMap mapping generated by Proguard if requested. could be null.
   */
  public static ProguardOutput createOptimizationActions(RuleContext ruleContext,
      FilesToRunProvider proguard,
      Artifact programJar,
      ImmutableList<Artifact> proguardSpecs,
      @Nullable Artifact proguardSeeds,
      @Nullable Artifact proguardUsage,
      @Nullable Artifact proguardMapping,
      Iterable<Artifact> libraryJars,
      Artifact proguardOutputJar,
      JavaSemantics semantics,
      @Nullable Integer optimizationPasses,
      @Nullable Artifact proguardOutputMap) throws InterruptedException {
    JavaOptimizationMode optMode = getJavaOptimizationMode(ruleContext);
    Preconditions.checkArgument(optMode != JavaOptimizationMode.NOOP);
    Preconditions.checkArgument(optMode != JavaOptimizationMode.LEGACY || !proguardSpecs.isEmpty());

    ProguardOutput output =
        getProguardOutputs(proguardOutputJar, proguardSeeds, proguardUsage, ruleContext, semantics,
            proguardOutputMap);


    if (Iterables.size(libraryJars) > 1) {
      JavaTargetAttributes attributes = new JavaTargetAttributes.Builder(semantics)
          .build();
      Artifact combinedLibraryJar = getProguardTempArtifact(ruleContext,
          optMode.name().toLowerCase(), "combined_library_jars.jar");
      new DeployArchiveBuilder(semantics, ruleContext)
          .setOutputJar(combinedLibraryJar)
          .setAttributes(attributes)
          .addRuntimeJars(libraryJars)
          .build();
      libraryJars = ImmutableList.of(combinedLibraryJar);
    }

    if (optimizationPasses == null) {
      // Run proguard as a single step.
      SpawnAction.Builder builder = defaultAction(
          proguard,
          programJar,
          proguardSpecs,
          proguardMapping,
          libraryJars,
          output.getOutputJar(),
          output.getMapping(),
          output.getProtoMapping(),
          output.getSeeds(),
          output.getUsage(),
          output.getConstantStringObfuscatedMapping(),
          output.getConfig())
          .setProgressMessage("Trimming binary with Proguard")
          .addOutput(proguardOutputJar);

      ruleContext.registerAction(builder.build(ruleContext));
    } else {
      // Optimization passes have been specified, so run proguard in multiple phases.
      Artifact lastStageOutput = getProguardTempArtifact(
          ruleContext, optMode.name().toLowerCase(), "proguard_preoptimization.jar");
      ruleContext.registerAction(
          defaultAction(
              proguard,
              programJar,
              proguardSpecs,
              proguardMapping,
              libraryJars,
              output.getOutputJar(),
              /* proguardOutputMap */ null,
              /* proguardOutputProtoMap */ null,
              output.getSeeds(),  // ProGuard only prints seeds during INITIAL and NORMAL runtypes.
              /* proguardUsage */ null,
              /* constantStringObfuscatedMapping */ null,
              /* proguardConfigOutput */ null)
              .setProgressMessage("Trimming binary with Proguard: Verification/Shrinking Pass")
              .addArgument("-runtype INITIAL")
              .addArgument("-nextstageoutput")
              .addOutputArgument(lastStageOutput)
              .build(ruleContext));

      for (int i = 1; i <= optimizationPasses; i++) {
        // Run configured optimizers in order in each pass
        for (Map.Entry<String, Optional<Label>> optimizer :
            getBytecodeOptimizers(ruleContext).entrySet()) {
          String mnemonic = optimizer.getKey();
          Optional<Label> target = optimizer.getValue();
          FilesToRunProvider executable = null;
          if (target.isPresent()) {
            for (TransitiveInfoCollection optimizerDep :
                ruleContext.getPrerequisites(":bytecode_optimizers", Mode.HOST)) {
              if (optimizerDep.getLabel().equals(target.get())) {
                executable = optimizerDep.getProvider(FilesToRunProvider.class);
                break;
              }
            }
          } else {
            checkState("Proguard".equals(mnemonic), "Need label to run %s", mnemonic);
            executable = proguard;
          }
          Artifact optimizationOutput = getProguardTempArtifact(
              ruleContext, optMode.name().toLowerCase(), mnemonic + "_optimization_" + i + ".jar");
          ruleContext.registerAction(
              defaultAction(
                      checkNotNull(executable, "couldn't find optimizer %s", optimizer),
                      programJar,
                      proguardSpecs,
                      proguardMapping,
                      libraryJars,
                      output.getOutputJar(),
                      /* proguardOutputMap */ null,
                      /* proguardOutputProtoMap */ null,
                      /* proguardSeeds */ null,
                      /* proguardUsage */ null,
                      /* constantStringObfuscatedMapping */ null,
                      /* proguardConfigOutput */ null)
                  .setProgressMessage("Trimming binary with %s: Optimization Pass %d", mnemonic, +i)
                  .setMnemonic(mnemonic)
                  .addArgument("-runtype OPTIMIZATION")
                  .addArgument("-laststageoutput")
                  .addInputArgument(lastStageOutput)
                  .addArgument("-nextstageoutput")
                  .addOutputArgument(optimizationOutput)
                  .build(ruleContext));
          lastStageOutput = optimizationOutput;
        }
      }

      SpawnAction.Builder builder = defaultAction(
          proguard,
          programJar,
          proguardSpecs,
          proguardMapping,
          libraryJars,
          output.getOutputJar(),
          output.getMapping(),
          output.getProtoMapping(),
          /* proguardSeeds */ null,  // runtype FINAL does not produce seeds.
          output.getUsage(),
          output.getConstantStringObfuscatedMapping(),
          output.getConfig())
          .setProgressMessage("Trimming binary with Proguard: Obfuscation and Final Output Pass")
          .addArgument("-runtype FINAL")
          .addArgument("-laststageoutput")
          .addInputArgument(lastStageOutput)
          .addOutput(proguardOutputJar);

      ruleContext.registerAction(builder.build(ruleContext));
    }

    return output;
  }

  private static SpawnAction.Builder defaultAction(
      FilesToRunProvider executable,
      Artifact programJar,
      ImmutableList<Artifact> proguardSpecs,
      @Nullable Artifact proguardMapping,
      Iterable<Artifact> libraryJars,
      Artifact proguardOutputJar,
      @Nullable Artifact proguardOutputMap,
      @Nullable Artifact proguardOutputProtoMap,
      @Nullable Artifact proguardSeeds,
      @Nullable Artifact proguardUsage,
      @Nullable Artifact constantStringObfuscatedMapping,
      @Nullable Artifact proguardConfigOutput) {

    Builder builder = new SpawnAction.Builder()
        .addInputs(libraryJars)
        .addInputs(proguardSpecs)
        .setExecutable(executable)
        .setMnemonic("Proguard")
        .addArgument("-forceprocessing")
        .addArgument("-injars")
        .addInputArgument(programJar)
        // This is handled by the build system there is no need for proguard to check if things are
        // up to date.
        .addArgument("-outjars")
        // Don't register the output jar as an output of the action, because multiple proguard
        // actions will be created for optimization runs which will overwrite the jar, and only
        // the final proguard action will declare the output jar as an output.
        .addArgument(proguardOutputJar.getExecPathString());

    for (Artifact libraryJar : libraryJars) {
      builder
          .addArgument("-libraryjars")
          .addArgument(libraryJar.getExecPathString());
    }

    if (proguardMapping != null) {
      builder
          .addArgument("-applymapping")
          .addInputArgument(proguardMapping);
    }

    for (Artifact proguardSpec : proguardSpecs) {
      builder.addArgument("@" + proguardSpec.getExecPathString());
    }

    if (proguardOutputMap != null) {
      builder
          .addArgument("-printmapping")
          .addOutputArgument(proguardOutputMap);
    }

    if (proguardOutputProtoMap != null) {
      builder
          .addArgument("-protomapping")
          .addOutputArgument(proguardOutputProtoMap);
    }

    if (constantStringObfuscatedMapping != null) {
      builder
          .addArgument("-obfuscatedconstantstringoutputfile")
          .addOutputArgument(constantStringObfuscatedMapping);
    }

    if (proguardSeeds != null) {
      builder
          .addArgument("-printseeds")
          .addOutputArgument(proguardSeeds);
    }

    if (proguardUsage != null) {
      builder
          .addArgument("-printusage")
          .addOutputArgument(proguardUsage);
    }

    if (proguardConfigOutput != null) {
      builder
          .addArgument("-printconfiguration")
          .addOutputArgument(proguardConfigOutput);
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

  private static Map<String, Optional<Label>> getBytecodeOptimizers(RuleContext ruleContext) {
    return ruleContext.getConfiguration().getFragment(JavaConfiguration.class)
        .getBytecodeOptimizers();
  }
}
