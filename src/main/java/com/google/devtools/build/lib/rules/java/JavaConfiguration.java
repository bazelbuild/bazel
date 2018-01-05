// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.common.options.TriState;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** A java compiler configuration containing the flags required for compilation. */
@AutoCodec
@Immutable
@SkylarkModule(
  name = "java",
  doc = "A java compiler configuration.",
  category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT
)
public final class JavaConfiguration extends Fragment {
  public static final ObjectCodec<JavaConfiguration> CODEC = new JavaConfiguration_AutoCodec();

  /** Values for the --java_classpath option */
  public enum JavaClasspathMode {
    /** Use full transitive classpaths, the default behavior. */
    OFF,
    /** JavaBuilder computes the reduced classpath before invoking javac. */
    JAVABUILDER,
  }

  /** Values for the --experimental_one_version_enforcement option */
  public enum OneVersionEnforcementLevel {
    /** Don't attempt to check for one version violations (the default) */
    OFF,
    /**
     * Check for one version violations, emit warnings to stderr if any are found, but don't break
     * the binary.
     */
    WARNING,
    /**
     * Check for one version violations, emit warnings to stderr if any are found, and break the
     * rule if it's found.
     */
    ERROR
  }

  /**
   * Values for the --java_optimization_mode option, which controls how Proguard is run over binary
   * and test targets. Note that for the moment this has no effect when building library targets.
   */
  public enum JavaOptimizationMode {
    /** Proguard is used iff top-level target has {@code proguard_specs} attribute. */
    LEGACY,
    /**
     * No link-time optimizations are applied, regardless of the top-level target's attributes. In
     * practice this mode skips Proguard completely, rather than invoking Proguard as a no-op.
     */
    NOOP("-dontshrink", "-dontoptimize", "-dontobfuscate"),
    /**
     * Symbols have different names except where configured not to rename.  This mode is primarily
     * intended to aid in identifying missing configuration directives that prevent symbols accessed
     * reflectively etc. from being renamed or removed.
     */
    RENAME("-dontshrink", "-dontoptimize"),
    /**
     * "Quickly" produce small binary typically without changing code structure.  In practice this
     * mode removes unreachable code and uses short symbol names except where configured not to
     * rename or remove.  This mode should build faster than {@link #OPTIMIZE_MINIFY} and may hence
     * be preferable during development.
     */
    FAST_MINIFY("-dontoptimize"),
    /**
     * Produce fully optimized binary with short symbol names and unreachable code removed.  Unlike
     * {@link #FAST_MINIFY}, this mode may apply code transformations, in addition to removing and
     * renaming code as the configuration allows, to produce a more compact binary.  This mode
     * should be preferable for producing and testing release binaries.
     */
    OPTIMIZE_MINIFY;

    private final String proguardDirectives;

    JavaOptimizationMode(String... donts) {
      StringBuilder proguardDirectives = new StringBuilder();
      for (String dont : donts) {
        checkArgument(dont.startsWith("-dont"), "invalid Proguard directive: %s", dont);
        proguardDirectives.append(dont).append('\n');
      }
      this.proguardDirectives = proguardDirectives.toString();
    }

    /**
     * Returns additional Proguard directives necessary for this mode (can be empty).
     */
    public String getImplicitProguardDirectives() {
      return proguardDirectives;
    }

    /**
     * Returns true if all affected targets should produce mappings from original to renamed symbol
     * names, regardless of the proguard_generate_mapping attribute.  This should be the case for
     * all modes that force symbols to be renamed.  By contrast, the {@link #NOOP} mode will never
     * produce a mapping file since no symbols are ever renamed.
     */
    public boolean alwaysGenerateOutputMapping() {
      switch (this) {
        case LEGACY:
        case NOOP:
          return false;
        case RENAME:
        case FAST_MINIFY:
        case OPTIMIZE_MINIFY:
          return true;
        default:
          throw new AssertionError("Unexpected mode: " + this);
      }
    }
  }

  private final ImmutableList<String> commandLineJavacFlags;
  private final Label javaLauncherLabel;
  private final boolean useIjars;
  private final boolean useHeaderCompilation;
  private final boolean headerCompilationDisableJavacFallback;
  private final boolean generateJavaDeps;
  private final boolean strictDepsJavaProtos;
  private final OneVersionEnforcementLevel enforceOneVersion;
  private final boolean enforceOneVersionOnJavaTests;
  private final boolean allowRuntimeDepsOnNeverLink;
  private final JavaClasspathMode javaClasspath;
  private final ImmutableList<String> defaultJvmFlags;
  private final ImmutableList<String> checkedConstraints;
  private final StrictDepsMode strictJavaDeps;
  private final Label proguardBinary;
  private final ImmutableList<Label> extraProguardSpecs;
  private final TriState bundleTranslations;
  private final ImmutableList<Label> translationTargets;
  private final JavaOptimizationMode javaOptimizationMode;
  private final ImmutableMap<String, Optional<Label>> bytecodeOptimizers;
  private final Label toolchainLabel;
  private final boolean explicitJavaTestDeps;
  private final boolean experimentalTestRunner;
  private final boolean jplPropagateCcLinkParamsStore;
  private final ImmutableList<Label> pluginList;

  // TODO(dmarting): remove once we have a proper solution for #2539
  private final boolean useLegacyBazelJavaTest;

  JavaConfiguration(
      boolean generateJavaDeps,
      List<String> defaultJvmFlags,
      JavaOptions javaOptions,
      Label toolchainLabel)
      throws InvalidConfigurationException {
    this.commandLineJavacFlags =
        ImmutableList.copyOf(JavaHelper.tokenizeJavaOptions(javaOptions.javacOpts));
    this.javaLauncherLabel = javaOptions.javaLauncher;
    this.useIjars = javaOptions.useIjars;
    this.useHeaderCompilation = javaOptions.headerCompilation;
    this.headerCompilationDisableJavacFallback = javaOptions.headerCompilationDisableJavacFallback;
    this.generateJavaDeps = generateJavaDeps;
    this.javaClasspath = javaOptions.javaClasspath;
    this.defaultJvmFlags = ImmutableList.copyOf(defaultJvmFlags);
    this.checkedConstraints = ImmutableList.copyOf(javaOptions.checkedConstraints);
    this.strictJavaDeps = javaOptions.strictJavaDeps;
    this.proguardBinary = javaOptions.proguard;
    this.extraProguardSpecs = ImmutableList.copyOf(javaOptions.extraProguardSpecs);
    this.bundleTranslations = javaOptions.bundleTranslations;
    this.toolchainLabel = toolchainLabel;
    this.javaOptimizationMode = javaOptions.javaOptimizationMode;
    this.useLegacyBazelJavaTest = javaOptions.legacyBazelJavaTest;
    this.strictDepsJavaProtos = javaOptions.strictDepsJavaProtos;
    this.enforceOneVersion = javaOptions.enforceOneVersion;
    this.enforceOneVersionOnJavaTests = javaOptions.enforceOneVersionOnJavaTests;
    this.allowRuntimeDepsOnNeverLink = javaOptions.allowRuntimeDepsOnNeverLink;
    this.explicitJavaTestDeps = javaOptions.explicitJavaTestDeps;
    this.experimentalTestRunner = javaOptions.experimentalTestRunner;
    this.jplPropagateCcLinkParamsStore = javaOptions.jplPropagateCcLinkParamsStore;

    ImmutableList.Builder<Label> translationsBuilder = ImmutableList.builder();
    for (String s : javaOptions.translationTargets) {
      try {
        Label label = Label.parseAbsolute(s);
        translationsBuilder.add(label);
      } catch (LabelSyntaxException e) {
        throw new InvalidConfigurationException("Invalid translations target '" + s + "', make " +
            "sure it uses correct absolute path syntax.", e);
      }
    }
    this.translationTargets = translationsBuilder.build();

    ImmutableMap.Builder<String, Optional<Label>> optimizersBuilder = ImmutableMap.builder();
    for (Map.Entry<String, Label> optimizer : javaOptions.bytecodeOptimizers.entrySet()) {
      String mnemonic = optimizer.getKey();
      if (optimizer.getValue() == null && !"Proguard".equals(mnemonic)) {
        throw new InvalidConfigurationException("Must supply label for optimizer " + mnemonic);
      }
      optimizersBuilder.put(mnemonic, Optional.fromNullable(optimizer.getValue()));
    }
    this.bytecodeOptimizers = optimizersBuilder.build();
    this.pluginList = ImmutableList.copyOf(javaOptions.pluginList);
  }

  @AutoCodec.Constructor
  JavaConfiguration(
      ImmutableList<String> commandLineJavacFlags,
      Label javaLauncherLabel,
      boolean useIjars,
      boolean useHeaderCompilation,
      boolean headerCompilationDisableJavacFallback,
      boolean generateJavaDeps,
      boolean strictDepsJavaProtos,
      OneVersionEnforcementLevel enforceOneVersion,
      boolean enforceOneVersionOnJavaTests,
      boolean allowRuntimeDepsOnNeverLink,
      JavaClasspathMode javaClasspath,
      ImmutableList<String> defaultJvmFlags,
      ImmutableList<String> checkedConstraints,
      StrictDepsMode strictJavaDeps,
      Label proguardBinary,
      ImmutableList<Label> extraProguardSpecs,
      TriState bundleTranslations,
      ImmutableList<Label> translationTargets,
      JavaOptimizationMode javaOptimizationMode,
      ImmutableMap<String, Optional<Label>> bytecodeOptimizers,
      Label toolchainLabel,
      boolean explicitJavaTestDeps,
      boolean experimentalTestRunner,
      boolean jplPropagateCcLinkParamsStore,
      ImmutableList<Label> pluginList,
      boolean useLegacyBazelJavaTest) {
    this.commandLineJavacFlags = commandLineJavacFlags;
    this.javaLauncherLabel = javaLauncherLabel;
    this.useIjars = useIjars;
    this.useHeaderCompilation = useHeaderCompilation;
    this.headerCompilationDisableJavacFallback = headerCompilationDisableJavacFallback;
    this.generateJavaDeps = generateJavaDeps;
    this.strictDepsJavaProtos = strictDepsJavaProtos;
    this.enforceOneVersion = enforceOneVersion;
    this.enforceOneVersionOnJavaTests = enforceOneVersionOnJavaTests;
    this.allowRuntimeDepsOnNeverLink = allowRuntimeDepsOnNeverLink;
    this.javaClasspath = javaClasspath;
    this.defaultJvmFlags = defaultJvmFlags;
    this.checkedConstraints = checkedConstraints;
    this.strictJavaDeps = strictJavaDeps;
    this.proguardBinary = proguardBinary;
    this.extraProguardSpecs = extraProguardSpecs;
    this.bundleTranslations = bundleTranslations;
    this.translationTargets = translationTargets;
    this.javaOptimizationMode = javaOptimizationMode;
    this.bytecodeOptimizers = bytecodeOptimizers;
    this.toolchainLabel = toolchainLabel;
    this.explicitJavaTestDeps = explicitJavaTestDeps;
    this.experimentalTestRunner = experimentalTestRunner;
    this.jplPropagateCcLinkParamsStore = jplPropagateCcLinkParamsStore;
    this.pluginList = pluginList;
    this.useLegacyBazelJavaTest = useLegacyBazelJavaTest;
  }

  @SkylarkCallable(name = "default_javac_flags", structField = true,
      doc = "The default flags for the Java compiler.")
  // TODO(bazel-team): this is the command-line passed options, we should remove from skylark
  // probably.
  public ImmutableList<String> getDefaultJavacFlags() {
    return commandLineJavacFlags;
  }

  @SkylarkCallable(
      name = "strict_java_deps",
      structField = true,
      doc = "The value of the strict_java_deps flag."
  )
  public String getStrictJavaDepsName() {
    return strictJavaDeps.name().toLowerCase();
  }

  @Override
  public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {
    if ((bundleTranslations == TriState.YES) && translationTargets.isEmpty()) {
      reporter.handle(Event.error("Translations enabled, but no message translations specified. " +
          "Use '--message_translations' to select the message translations to use"));
    }
  }

  @Override
  public void addGlobalMakeVariables(Builder<String, String> globalMakeEnvBuilder) {
    globalMakeEnvBuilder.put("JAVA_TRANSLATIONS", buildTranslations() ? "1" : "0");
  }

  @Override
  public boolean compatibleWithStrategy(String strategyName) {
    if (strategyName.equals("experimental_worker")) {
      return explicitJavaTestDeps() && useExperimentalTestRunner();
    }
    return true;
  }

  /**
   * Returns true iff Java compilation should use ijars.
   */
  public boolean getUseIjars() {
    return useIjars;
  }

  /** Returns true iff Java header compilation is enabled. */
  public boolean useHeaderCompilation() {
    return useHeaderCompilation;
  }

  /**
   * If --java_header_compilation is set, report diagnostics from turbine instead of falling back to
   * javac. Diagnostics will be produced more quickly, but may be less helpful.
   */
  public boolean headerCompilationDisableJavacFallback() {
    return headerCompilationDisableJavacFallback;
  }

  /**
   * Returns true iff dependency information is generated after compilation.
   */
  public boolean getGenerateJavaDeps() {
    return generateJavaDeps;
  }

  public JavaClasspathMode getReduceJavaClasspath() {
    return javaClasspath;
  }

  public ImmutableList<String> getDefaultJvmFlags() {
    return defaultJvmFlags;
  }

  public ImmutableList<String> getCheckedConstraints() {
    return checkedConstraints;
  }

  public StrictDepsMode getStrictJavaDeps() {
    return strictJavaDeps;
  }

  public StrictDepsMode getFilteredStrictJavaDeps() {
    StrictDepsMode strict = getStrictJavaDeps();
    switch (strict) {
      case STRICT:
      case DEFAULT:
        return StrictDepsMode.ERROR;
      default:   // OFF, WARN, ERROR
        return strict;
    }
  }

  /**
   * @return proper label only if --java_launcher= is specified, otherwise null.
   */
  public Label getJavaLauncherLabel() {
    return javaLauncherLabel;
  }

  /**
   * Returns the label provided with --proguard_top, if any.
   */
  @Nullable
  public Label getProguardBinary() {
    return proguardBinary;
  }

  /**
   * Returns all labels provided with --extra_proguard_specs.
   */
  public ImmutableList<Label> getExtraProguardSpecs() {
    return extraProguardSpecs;
  }

  /**
   * Returns the raw translation targets.
   */
  public ImmutableList<Label> getTranslationTargets() {
    return translationTargets;
  }

  /**
   * Returns true if the we should build translations.
   */
  public boolean buildTranslations() {
    return (bundleTranslations != TriState.NO) && !translationTargets.isEmpty();
  }

  /**
   * Returns whether translations were explicitly disabled.
   */
  public boolean isTranslationsDisabled() {
    return bundleTranslations == TriState.NO;
  }

  /**
   * Returns the label of the default java_toolchain rule
   */
  public Label getToolchainLabel() {
    return toolchainLabel;
  }

  /**
   * Returns the --java_optimization_mode flag setting. Note that running with a different mode over
   * the same binary or test target typically invalidates the cached output Jar for that target,
   * but since Proguard doesn't run on libraries, the outputs for library targets remain valid.
   */
  public JavaOptimizationMode getJavaOptimizationMode() {
    return javaOptimizationMode;
  }

  /**
   * Returns ordered list of optimizers to run.
   */
  public ImmutableMap<String, Optional<Label>> getBytecodeOptimizers() {
    return bytecodeOptimizers;
  }

  /**
   * Returns true if java_test in Bazel should behave in legacy mode that existed before we
   * open-sourced our test runner.
   */
  public boolean useLegacyBazelJavaTest() {
    return useLegacyBazelJavaTest;
  }

  /**
   * Returns true if we should be the ExperimentalTestRunner instead of the BazelTestRunner for
   * bazel's java_test runs.
   */
  public boolean useExperimentalTestRunner() {
    return experimentalTestRunner;
  }

  /**
   * Make it mandatory for java_test targets to explicitly declare any JUnit or Hamcrest
   * dependencies instead of accidentally obtaining them from the TestRunner's dependencies.
   */
  public boolean explicitJavaTestDeps() {
    return explicitJavaTestDeps;
  }

  /**
   * Returns an enum representing whether or not Bazel should attempt to enforce one-version
   * correctness on java_binary rules using the 'oneversion' tool in the java_toolchain.
   *
   * One-version correctness will inspect for multiple non-identical versions of java classes in the
   * transitive dependencies for a java_binary.
   */
  public OneVersionEnforcementLevel oneVersionEnforcementLevel() {
    return enforceOneVersion;
  }

  public boolean enforceOneVersionOnJavaTests() {
    return enforceOneVersionOnJavaTests;
  }

  public boolean getAllowRuntimeDepsOnNeverLink() {
    return allowRuntimeDepsOnNeverLink;
  }

  public boolean strictDepsJavaProtos() {
    return strictDepsJavaProtos;
  }

  public boolean jplPropagateCcLinkParamsStore() {
    return jplPropagateCcLinkParamsStore;
  }

  public List<Label> getPlugins() {
    return pluginList;
  }
}
