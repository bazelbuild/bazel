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

import static com.google.devtools.build.lib.util.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;

import java.util.List;

import javax.annotation.Nullable;

/**
 * Information about the JDK used by the <code>java_*</code> rules.
 */
@Immutable
public final class JavaToolchainProvider implements TransitiveInfoProvider {

  /**
   * Constructs the list of javac options.
   *
   * @param ruleContext The rule context of the current rule.
   * @return the list of flags provided by the {@code java_toolchain} rule merged with the one
   *         provided by the {@link JavaConfiguration} fragment.
   */
  public static List<String> getDefaultJavacOptions(RuleContext ruleContext) {
    JavaToolchainProvider javaToolchain =
        ruleContext.getPrerequisite(":java_toolchain", Mode.TARGET, JavaToolchainProvider.class);
    if (javaToolchain == null) {
      ruleContext.ruleError("The --java_toolchain option does not point to a java_toolchain rule.");
      return ImmutableList.of();
    }
    return javaToolchain.getJavacOptions();
  }

  /**
   * Constructs the list of options to pass to the JVM running the java compiler.
   *
   * @param ruleContext The rule context of the current rule.
   * @return the list of flags provided by the {@code java_toolchain} rule merged with the one
   *         provided by the {@link JavaConfiguration} fragment.
   */
  public static List<String> getDefaultJavacJvmOptions(RuleContext ruleContext) {
    if (!ruleContext.getRule().isAttrDefined(":java_toolchain", BuildType.LABEL)) {
      // As some rules might not have java_toolchain dependency (e.g., java_import), we silently
      // ignore it. The rules needing it will error in #getDefaultJavacOptions(RuleContext) anyway.
      return ImmutableList.of();
    }
    JavaToolchainProvider javaToolchain =
        ruleContext.getPrerequisite(":java_toolchain", Mode.TARGET, JavaToolchainProvider.class);
    if (javaToolchain == null) {
      ruleContext.ruleError("The --java_toolchain option does not point to a java_toolchain rule.");
      return ImmutableList.of();
    }
    return javaToolchain.getJavacJvmOptions();
  }

  /** Returns the {@link Artifact} of the header compiler deploy jar. */
  public static Artifact getHeaderCompilerJar(RuleContext ruleContext) {
    JavaToolchainProvider javaToolchain =
        ruleContext.getPrerequisite(":java_toolchain", Mode.TARGET, JavaToolchainProvider.class);
    if (javaToolchain == null) {
      ruleContext.ruleError("The --java_toolchain option does not point to a java_toolchain rule.");
      return null;
    }
    return javaToolchain.getHeaderCompiler();
  }

  /**
   * Constructs the compilation bootclasspath.
   *
   * @param ruleContext The rule context of the current rule.
   */
  public static NestedSet<Artifact> getDefaultBootclasspath(RuleContext ruleContext) {
    JavaToolchainProvider javaToolchain =
        ruleContext.getPrerequisite(":java_toolchain", Mode.TARGET, JavaToolchainProvider.class);
    if (javaToolchain == null) {
      ruleContext.ruleError("The --java_toolchain option does not point to a java_toolchain rule.");
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return javaToolchain.getBootclasspath();
  }

  /**
   * Constructs the compilation extclasspath.
   *
   * @param ruleContext The rule context of the current rule.
   */
  public static NestedSet<Artifact> getDefaultExtclasspath(RuleContext ruleContext) {
    JavaToolchainProvider javaToolchain =
        ruleContext.getPrerequisite(":java_toolchain", Mode.TARGET, JavaToolchainProvider.class);
    if (javaToolchain == null) {
      ruleContext.ruleError("The --java_toolchain option does not point to a java_toolchain rule.");
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return javaToolchain.getExtclasspath();
  }

  private final String sourceVersion;
  private final String targetVersion;
  @Nullable private final NestedSet<Artifact> bootclasspath;
  @Nullable private final NestedSet<Artifact> extclasspath;
  private final String encoding;
  private final ImmutableList<String> javacOptions;
  private final ImmutableList<String> javacJvmOptions;
  @Nullable private final Artifact headerCompiler;

  public JavaToolchainProvider(
      JavaToolchainData data,
      @Nullable NestedSet<Artifact> bootclasspath,
      @Nullable NestedSet<Artifact> extclasspath,
      List<String> defaultJavacFlags,
      List<String> defaultJavacJvmOpts,
      @Nullable Artifact headerCompiler) {
    this.sourceVersion = checkNotNull(data.getSourceVersion(), "sourceVersion must not be null");
    this.targetVersion = checkNotNull(data.getTargetVersion(), "targetVersion must not be null");
    this.bootclasspath = bootclasspath;
    this.extclasspath = extclasspath;
    this.encoding = checkNotNull(data.getEncoding(), "encoding must not be null");
    this.headerCompiler = headerCompiler;

    // merges the defaultJavacFlags from
    // {@link JavaConfiguration} with the flags from the {@code java_toolchain} rule.
    this.javacOptions =
        ImmutableList.<String>builder()
            .addAll(data.getJavacOptions())
            .addAll(defaultJavacFlags)
            .build();
    // merges the defaultJavaBuilderJvmFlags from
    // {@link JavaConfiguration} with the flags from the {@code java_toolchain} rule.
    this.javacJvmOptions =
        ImmutableList.<String>builder()
            .addAll(data.getJavacJvmOptions())
            .addAll(defaultJavacJvmOpts)
            .build();
  }

  /** @return the list of default options for the java compiler */
  public ImmutableList<String> getJavacOptions() {
    return javacOptions;
  }

  /** @return the list of default options for the JVM running the java compiler */
  public ImmutableList<String> getJavacJvmOptions() {
    return javacJvmOptions;
  }

  /** @return the input Java language level */
  public String getSourceVersion() {
    return sourceVersion;
  }

  /** @return the target Java language level */
  public String getTargetVersion() {
    return targetVersion;
  }

  /** @return the target Java bootclasspath */
  public NestedSet<Artifact> getBootclasspath() {
    return bootclasspath;
  }

  /** @return the target Java extclasspath */
  public NestedSet<Artifact> getExtclasspath() {
    return extclasspath;
  }

  /** @return the encoding for Java source files */
  @Nullable
  public String getEncoding() {
    return encoding;
  }

  /** @return the {@link Artifact} of the Header Compiler deploy jar */
  @Nullable
  public Artifact getHeaderCompiler() {
    return headerCompiler;
  }
}
