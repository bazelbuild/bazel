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

import com.google.common.collect.ImmutableList;
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
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.common.options.TriState;

import java.util.List;

import javax.annotation.Nullable;

/**
 * A java compiler configuration containing the flags required for compilation.
 */
@Immutable
@SkylarkModule(name = "java", doc = "A java compiler configuration")
public final class JavaConfiguration extends Fragment {
  /**
   * Values for the --experimental_java_classpath option
   */
  public static enum JavaClasspathMode {
    /** Use full transitive classpaths, the default behavior. */
    OFF,
    /** JavaBuilder computes the reduced classpath before invoking javac. */
    JAVABUILDER,
    /** Blaze computes the reduced classpath before invoking JavaBuilder. */
    BLAZE
  }

  /**
   * Values for the --java_optimization_mode option, which controls how Proguard is run over binary
   * and test targets.  Note that for the moment this has no effect when building library targets.
   */
  public static enum JavaOptimizationMode {
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

    private String proguardDirectives;

    private JavaOptimizationMode(String... donts) {
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
  private final Label javaBuilderTop;
  private final ImmutableList<String> defaultJavaBuilderJvmOpts;
  private final Label javaLangtoolsJar;
  private final boolean useIjars;
  private final boolean useHeaderCompilation;
  private final boolean generateJavaDeps;
  private final JavaClasspathMode experimentalJavaClasspath;
  private final ImmutableList<String> javaWarns;
  private final ImmutableList<String> defaultJvmFlags;
  private final ImmutableList<String> checkedConstraints;
  private final StrictDepsMode strictJavaDeps;
  private final Label javacBootclasspath;
  private final Label javacExtdir;
  private final ImmutableList<String> javacOpts;
  private final Label proguardBinary;
  private final ImmutableList<Label> extraProguardSpecs;
  private final TriState bundleTranslations;
  private final ImmutableList<Label> translationTargets;
  private final String javaCpu;
  private final JavaOptimizationMode javaOptimizationMode;
  private final Label javaToolchain;

  // TODO(dmarting): remove when we have rolled out the new behavior
  private final boolean legacyBazelJavaTest;

  JavaConfiguration(boolean generateJavaDeps,
      List<String> defaultJvmFlags, JavaOptions javaOptions, Label javaToolchain, String javaCpu,
      ImmutableList<String> defaultJavaBuilderJvmOpts)
          throws InvalidConfigurationException {
    this.commandLineJavacFlags =
        ImmutableList.copyOf(JavaHelper.tokenizeJavaOptions(javaOptions.javacOpts));
    this.javaLauncherLabel = javaOptions.javaLauncher;
    this.javaBuilderTop = javaOptions.javaBuilderTop;
    this.defaultJavaBuilderJvmOpts = defaultJavaBuilderJvmOpts;
    this.javaLangtoolsJar = javaOptions.javaLangtoolsJar;
    this.useIjars = javaOptions.useIjars;
    this.useHeaderCompilation = javaOptions.headerCompilation;
    this.generateJavaDeps = generateJavaDeps;
    this.experimentalJavaClasspath = javaOptions.experimentalJavaClasspath;
    this.javaWarns = ImmutableList.copyOf(javaOptions.javaWarns);
    this.defaultJvmFlags = ImmutableList.copyOf(defaultJvmFlags);
    this.checkedConstraints = ImmutableList.copyOf(javaOptions.checkedConstraints);
    this.strictJavaDeps = javaOptions.strictJavaDeps;
    this.javacBootclasspath = javaOptions.javacBootclasspath;
    this.javacExtdir = javaOptions.javacExtdir;
    this.javacOpts = ImmutableList.copyOf(javaOptions.javacOpts);
    this.proguardBinary = javaOptions.proguard;
    this.extraProguardSpecs = ImmutableList.copyOf(javaOptions.extraProguardSpecs);
    this.bundleTranslations = javaOptions.bundleTranslations;
    this.javaCpu = javaCpu;
    this.javaToolchain = javaToolchain;
    this.javaOptimizationMode = javaOptions.javaOptimizationMode;
    this.legacyBazelJavaTest = javaOptions.legacyBazelJavaTest;

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
  }

  @SkylarkCallable(name = "default_javac_flags", structField = true,
      doc = "The default flags for the Java compiler.")
  // TODO(bazel-team): this is the command-line passed options, we should remove from skylark
  // probably.
  public ImmutableList<String> getDefaultJavacFlags() {
    return commandLineJavacFlags;
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
    globalMakeEnvBuilder.put("JAVA_CPU", javaCpu);
  }

  /**
   * Returns the default javabuilder jar
   */
  public Label getDefaultJavaBuilderJar() {
    return javaBuilderTop;
  }

  /**
   * Returns the default JVM flags to be used when invoking javabuilder.
   */
  public ImmutableList<String> getDefaultJavaBuilderJvmFlags() {
    return defaultJavaBuilderJvmOpts;
  }

  /**
   * Returns the default java langtools jar
   */
  public Label getDefaultJavaLangtoolsJar() {
    return javaLangtoolsJar;
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
   * Returns true iff dependency information is generated after compilation.
   */
  public boolean getGenerateJavaDeps() {
    return generateJavaDeps;
  }

  public JavaClasspathMode getReduceJavaClasspath() {
    return experimentalJavaClasspath;
  }

  /**
   * Returns the extra warnings enabled for Java compilation.
   */
  public ImmutableList<String> getJavaWarns() {
    return javaWarns;
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

  public Label getJavacBootclasspath() {
    return javacBootclasspath;
  }

  public Label getJavacExtdir() {
    return javacExtdir;
  }

  public ImmutableList<String> getJavacOpts() {
    return javacOpts;
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
    return javaToolchain;
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
   * Returns true if java_test in Bazel should behave in legacy mode that existed before we
   * open-sourced our test runner.
   */
  public boolean useLegacyBazelJavaTest() {
    return legacyBazelJavaTest;
  }
}
