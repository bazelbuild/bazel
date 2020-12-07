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

import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import com.google.common.base.Ascii;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.analysis.starlark.annotations.StarlarkConfigurationField;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaConfigurationApi;
import com.google.devtools.common.options.TriState;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/** A java compiler configuration containing the flags required for compilation. */
@Immutable
@RequiresOptions(options = {JavaOptions.class, PlatformOptions.class})
public final class JavaConfiguration extends Fragment implements JavaConfigurationApi {

  /** Values for the --java_classpath option */
  public enum JavaClasspathMode {
    /** Use full transitive classpaths, the default behavior. */
    OFF,
    /** JavaBuilder computes the reduced classpath before invoking javac. */
    JAVABUILDER,
    /** Bazel computes the reduced classpath and tries it in a separate action invocation. */
    BAZEL
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

  /** Values for the --experimental_import_deps_checking option */
  public enum ImportDepsCheckingLevel {
    /** Turn off the import_deps checking. */
    OFF,
    /** Emit warnings when the dependencies of java_import/aar_import are not complete. */
    WARNING,
    /** Emit errors when the dependencies of java_import/aar_import are not complete. */
    ERROR
  }

  private final ImmutableList<String> commandLineJavacFlags;
  private final Label javaLauncherLabel;
  private final boolean useIjars;
  private final boolean useHeaderCompilation;
  private final boolean generateJavaDeps;
  private final boolean strictDepsJavaProtos;
  private final boolean isDisallowStrictDepsForJpl;
  private final OneVersionEnforcementLevel enforceOneVersion;
  private final boolean enforceOneVersionOnJavaTests;
  private final ImportDepsCheckingLevel importDepsCheckingLevel;
  private final boolean allowRuntimeDepsOnNeverLink;
  private final JavaClasspathMode javaClasspath;
  private final boolean inmemoryJdepsFiles;
  private final ImmutableList<String> defaultJvmFlags;
  private final ImmutableList<String> checkedConstraints;
  private final StrictDepsMode strictJavaDeps;
  private final String fixDepsTool;
  private final Label proguardBinary;
  private final ImmutableList<Label> extraProguardSpecs;
  private final TriState bundleTranslations;
  private final ImmutableList<Label> translationTargets;
  private final NamedLabel bytecodeOptimizer;
  private final boolean splitBytecodeOptimizationPass;
  private final boolean enforceProguardFileExtension;
  private final Label toolchainLabel;
  private final Label runtimeLabel;
  private final boolean explicitJavaTestDeps;
  private final boolean jplPropagateCcLinkParamsStore;
  private final boolean addTestSupportToCompileTimeDeps;
  private final boolean isJlplStrictDepsEnforced;
  private final ImmutableList<Label> pluginList;
  private final boolean requireJavaToolchainHeaderCompilerDirect;
  private final boolean disallowResourceJars;
  private final boolean disallowLegacyJavaToolchainFlags;
  private final boolean experimentalTurbineAnnotationProcessing;

  // TODO(dmarting): remove once we have a proper solution for #2539
  private final boolean useLegacyBazelJavaTest;

  public JavaConfiguration(BuildOptions buildOptions) throws InvalidConfigurationException {
    JavaOptions javaOptions = buildOptions.get(JavaOptions.class);
    this.commandLineJavacFlags =
        ImmutableList.copyOf(JavaHelper.tokenizeJavaOptions(javaOptions.javacOpts));
    this.javaLauncherLabel = javaOptions.javaLauncher;
    this.useIjars = javaOptions.useIjars;
    this.useHeaderCompilation = javaOptions.headerCompilation;
    this.generateJavaDeps =
        javaOptions.javaDeps || javaOptions.javaClasspath != JavaClasspathMode.OFF;
    this.javaClasspath = javaOptions.javaClasspath;
    this.inmemoryJdepsFiles = javaOptions.inmemoryJdepsFiles;
    this.defaultJvmFlags = ImmutableList.copyOf(javaOptions.jvmOpts);
    this.checkedConstraints = ImmutableList.copyOf(javaOptions.checkedConstraints);
    this.strictJavaDeps = javaOptions.strictJavaDeps;
    this.fixDepsTool = javaOptions.fixDepsTool;
    this.proguardBinary = javaOptions.proguard;
    this.extraProguardSpecs = ImmutableList.copyOf(javaOptions.extraProguardSpecs);
    this.splitBytecodeOptimizationPass = javaOptions.splitBytecodeOptimizationPass;
    this.enforceProguardFileExtension = javaOptions.enforceProguardFileExtension;
    this.bundleTranslations = javaOptions.bundleTranslations;
    this.toolchainLabel = javaOptions.javaToolchain;
    this.runtimeLabel = javaOptions.javaBase;
    this.useLegacyBazelJavaTest = javaOptions.legacyBazelJavaTest;
    this.strictDepsJavaProtos = javaOptions.strictDepsJavaProtos;
    this.isDisallowStrictDepsForJpl = javaOptions.isDisallowStrictDepsForJpl;
    this.enforceOneVersion = javaOptions.enforceOneVersion;
    this.enforceOneVersionOnJavaTests = javaOptions.enforceOneVersionOnJavaTests;
    this.importDepsCheckingLevel = javaOptions.importDepsCheckingLevel;
    this.allowRuntimeDepsOnNeverLink = javaOptions.allowRuntimeDepsOnNeverLink;
    this.explicitJavaTestDeps = javaOptions.explicitJavaTestDeps;
    this.jplPropagateCcLinkParamsStore = javaOptions.jplPropagateCcLinkParamsStore;
    this.isJlplStrictDepsEnforced = javaOptions.isJlplStrictDepsEnforced;
    this.disallowResourceJars = javaOptions.disallowResourceJars;
    this.addTestSupportToCompileTimeDeps = javaOptions.addTestSupportToCompileTimeDeps;

    ImmutableList.Builder<Label> translationsBuilder = ImmutableList.builder();
    for (String s : javaOptions.translationTargets) {
      try {
        Label label = Label.parseAbsolute(s, ImmutableMap.of());
        translationsBuilder.add(label);
      } catch (LabelSyntaxException e) {
        throw new InvalidConfigurationException(
            "Invalid translations target '"
                + s
                + "', make "
                + "sure it uses correct absolute path syntax.",
            e);
      }
    }
    this.translationTargets = translationsBuilder.build();

    Map<String, Label> optimizers = javaOptions.bytecodeOptimizers;
    checkState(
        optimizers.size() <= 1,
        "--experimental_bytecode_optimizers can only accept up to one mapping, but %s mappings "
            + "were provided.",
        optimizers.size());
    Map.Entry<String, Label> optimizer = Iterables.getOnlyElement(optimizers.entrySet());
    String mnemonic = optimizer.getKey();
    Label optimizerLabel = optimizer.getValue();
    if (optimizerLabel == null && !"Proguard".equals(mnemonic)) {
      throw new InvalidConfigurationException("Must supply label for optimizer " + mnemonic);
    }
    this.bytecodeOptimizer = NamedLabel.create(mnemonic, Optional.fromNullable(optimizerLabel));

    this.pluginList = ImmutableList.copyOf(javaOptions.pluginList);
    this.requireJavaToolchainHeaderCompilerDirect =
        javaOptions.requireJavaToolchainHeaderCompilerDirect;
    this.disallowLegacyJavaToolchainFlags = javaOptions.disallowLegacyJavaToolchainFlags;
    this.experimentalTurbineAnnotationProcessing =
        javaOptions.experimentalTurbineAnnotationProcessing;

    if (javaOptions.disallowLegacyJavaToolchainFlags) {
      checkLegacyToolchainFlagIsUnset(
          "javabase", javaOptions.javaBase, javaOptions.defaultJavaBase());
      checkLegacyToolchainFlagIsUnset(
          "host_javabase", javaOptions.getHostJavaBase(), javaOptions.defaultHostJavaBase());
      checkLegacyToolchainFlagIsUnset(
          "java_toolchain", javaOptions.javaToolchain, javaOptions.defaultJavaToolchain());
      checkLegacyToolchainFlagIsUnset(
          "host_java_toolchain", javaOptions.hostJavaToolchain, javaOptions.defaultJavaToolchain());
    }
  }

  private static void checkLegacyToolchainFlagIsUnset(String flag, Label label, Label defaultValue)
      throws InvalidConfigurationException {
    if (!Objects.equals(label, defaultValue)) {
      throw new InvalidConfigurationException(
          String.format(
              "--%s=%s is no longer supported, use --platforms instead (see #7849)", flag, label));
    }
  }

  @Override
  // TODO(bazel-team): this is the command-line passed options, we should remove from Starlark
  // probably.
  public ImmutableList<String> getDefaultJavacFlags() {
    return commandLineJavacFlags;
  }

  @Override
  public String getStrictJavaDepsName() {
    return Ascii.toLowerCase(strictJavaDeps.name());
  }

  @Override
  public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {
    if ((bundleTranslations == TriState.YES) && translationTargets.isEmpty()) {
      reporter.handle(
          Event.error(
              "Translations enabled, but no message translations specified. "
                  + "Use '--message_translations' to select the message translations to use"));
    }
  }

  /** Returns true iff Java compilation should use ijars. */
  public boolean getUseIjars() {
    return useIjars;
  }

  /** Returns true iff Java header compilation is enabled. */
  public boolean useHeaderCompilation() {
    return useHeaderCompilation;
  }

  /** Returns true iff dependency information is generated after compilation. */
  public boolean getGenerateJavaDeps() {
    return generateJavaDeps;
  }

  public JavaClasspathMode getReduceJavaClasspath() {
    return javaClasspath;
  }

  public boolean inmemoryJdepsFiles() {
    return inmemoryJdepsFiles;
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
      default: // OFF, WARN, ERROR
        return strict;
    }
  }

  /** Which tool to use for fixing dependency errors. */
  public String getFixDepsTool() {
    return fixDepsTool;
  }

  /** @return proper label only if --java_launcher= is specified, otherwise null. */
  public Label getJavaLauncherLabel() {
    return javaLauncherLabel;
  }

  /** Returns the label provided with --proguard_top, if any. */
  @StarlarkConfigurationField(
      name = "proguard_top",
      doc = "Returns the label provided with --proguard_top, if any.",
      defaultInToolRepository = true)
  @Nullable
  public Label getProguardBinary() {
    return proguardBinary;
  }

  /** Returns all labels provided with --extra_proguard_specs. */
  public ImmutableList<Label> getExtraProguardSpecs() {
    return extraProguardSpecs;
  }

  /**
   * Returns whether the OPTIMIZATION stage of the bytecode optimizer will be split across multiple
   * actions.
   */
  public boolean splitBytecodeOptimizationPass() {
    return splitBytecodeOptimizationPass;
  }

  /** Returns whether ProGuard configuration files are required to use a *.pgcfg extension. */
  public boolean enforceProguardFileExtension() {
    return enforceProguardFileExtension;
  }

  /** Returns the raw translation targets. */
  public ImmutableList<Label> getTranslationTargets() {
    return translationTargets;
  }

  /** Returns true if the we should build translations. */
  public boolean buildTranslations() {
    return (bundleTranslations != TriState.NO) && !translationTargets.isEmpty();
  }

  /** Returns whether translations were explicitly disabled. */
  public boolean isTranslationsDisabled() {
    return bundleTranslations == TriState.NO;
  }

  /** Returns the label of the default java_toolchain rule */
  @StarlarkConfigurationField(
      name = "java_toolchain",
      doc = "Returns the label of the default java_toolchain rule.",
      defaultLabel = "//tools/jdk:toolchain",
      defaultInToolRepository = true)
  public Label getToolchainLabel() {
    if (disallowLegacyJavaToolchainFlags) {
      throw new IllegalStateException("--java_toolchain is no longer supported");
    }
    return toolchainLabel;
  }

  /** Returns the label of the {@code java_runtime} rule representing the JVM in use. */
  public Label getRuntimeLabel() {
    return runtimeLabel;
  }

  /** Stores a String name and an optional associated label. */
  @AutoValue
  public abstract static class NamedLabel {
    public static NamedLabel create(String name, Optional<Label> label) {
      return new AutoValue_JavaConfiguration_NamedLabel(name, label);
    }

    public abstract String name();

    public abstract Optional<Label> label();
  }

  /** Returns ordered list of optimizers to run. */
  public NamedLabel getBytecodeOptimizer() {
    return bytecodeOptimizer;
  }

  /**
   * Returns true if java_test in Bazel should behave in legacy mode that existed before we
   * open-sourced our test runner.
   */
  public boolean useLegacyBazelJavaTest() {
    return useLegacyBazelJavaTest;
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
   * <p>One-version correctness will inspect for multiple non-identical versions of java classes in
   * the transitive dependencies for a java_binary.
   */
  public OneVersionEnforcementLevel oneVersionEnforcementLevel() {
    return enforceOneVersion;
  }

  public boolean enforceOneVersionOnJavaTests() {
    return enforceOneVersionOnJavaTests;
  }

  public ImportDepsCheckingLevel getImportDepsCheckingLevel() {
    return importDepsCheckingLevel;
  }

  public boolean getAllowRuntimeDepsOnNeverLink() {
    return allowRuntimeDepsOnNeverLink;
  }

  public boolean strictDepsJavaProtos() {
    return strictDepsJavaProtos;
  }

  public boolean isDisallowStrictDepsForJpl() {
    return isDisallowStrictDepsForJpl;
  }

  public boolean jplPropagateCcLinkParamsStore() {
    return jplPropagateCcLinkParamsStore;
  }

  public boolean addTestSupportToCompileTimeDeps() {
    return addTestSupportToCompileTimeDeps;
  }

  public boolean isJlplStrictDepsEnforced() {
    return isJlplStrictDepsEnforced;
  }

  @Override
  public ImmutableList<Label> getPlugins() {
    return pluginList;
  }

  public boolean requireJavaToolchainHeaderCompilerDirect() {
    return requireJavaToolchainHeaderCompilerDirect;
  }

  public boolean disallowResourceJars() {
    return disallowResourceJars;
  }

  public boolean experimentalTurbineAnnotationProcessing() {
    return experimentalTurbineAnnotationProcessing;
  }
}
