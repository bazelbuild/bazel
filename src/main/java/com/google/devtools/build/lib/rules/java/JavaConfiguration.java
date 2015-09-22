// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.syntax.SkylarkCallable;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.common.options.TriState;

import java.util.List;

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

  private final ImmutableList<String> commandLineJavacFlags;
  private final Label javaLauncherLabel;
  private final Label javaBuilderTop;
  private final ImmutableList<String> defaultJavaBuilderJvmOpts;
  private final Label javaLangtoolsJar;
  private final boolean useIjars;
  private final boolean generateJavaDeps;
  private final JavaClasspathMode experimentalJavaClasspath;
  private final ImmutableList<String> javaWarns;
  private final ImmutableList<String> defaultJvmFlags;
  private final ImmutableList<String> checkedConstraints;
  private final StrictDepsMode strictJavaDeps;
  private final Label javacBootclasspath;
  private final Label javacExtdir;
  private final ImmutableList<String> javacOpts;
  private final TriState bundleTranslations;
  private final ImmutableList<Label> translationTargets;
  private final String javaCpu;
  private final boolean allowPrecompiledJarsInSrcs;

  private Label javaToolchain;

  JavaConfiguration(boolean generateJavaDeps,
      List<String> defaultJvmFlags, JavaOptions javaOptions, Label javaToolchain, String javaCpu,
      ImmutableList<String> defaultJavaBuilderJvmOpts) throws InvalidConfigurationException {
    this.commandLineJavacFlags =
        ImmutableList.copyOf(JavaHelper.tokenizeJavaOptions(javaOptions.javacOpts));
    this.javaLauncherLabel = javaOptions.javaLauncher;
    this.javaBuilderTop = javaOptions.javaBuilderTop;
    this.defaultJavaBuilderJvmOpts = defaultJavaBuilderJvmOpts;
    this.javaLangtoolsJar = javaOptions.javaLangtoolsJar;
    this.useIjars = javaOptions.useIjars;
    this.generateJavaDeps = generateJavaDeps;
    this.experimentalJavaClasspath = javaOptions.experimentalJavaClasspath;
    this.javaWarns = ImmutableList.copyOf(javaOptions.javaWarns);
    this.defaultJvmFlags = ImmutableList.copyOf(defaultJvmFlags);
    this.checkedConstraints = ImmutableList.copyOf(javaOptions.checkedConstraints);
    this.strictJavaDeps = javaOptions.strictJavaDeps;
    this.javacBootclasspath = javaOptions.javacBootclasspath;
    this.javacExtdir = javaOptions.javacExtdir;
    this.javacOpts = ImmutableList.copyOf(javaOptions.javacOpts);
    this.bundleTranslations = javaOptions.bundleTranslations;
    this.javaCpu = javaCpu;
    this.javaToolchain = javaToolchain;
    this.allowPrecompiledJarsInSrcs = javaOptions.allowPrecompiledJarsInSrcs;

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
  public List<String> getDefaultJavacFlags() {
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
  public List<String> getJavaWarns() {
    return javaWarns;
  }

  public List<String> getDefaultJvmFlags() {
    return defaultJvmFlags;
  }

  public List<String> getCheckedConstraints() {
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

  public List<String> getJavacOpts() {
    return javacOpts;
  }

  /**
   * Returns the raw translation targets.
   */
  public List<Label> getTranslationTargets() {
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

  /** Returns whether pre-compiled jar files should be allowed in srcs. */
  public boolean allowPrecompiledJarsInSrcs() {
    return allowPrecompiledJarsInSrcs;
  }
}
