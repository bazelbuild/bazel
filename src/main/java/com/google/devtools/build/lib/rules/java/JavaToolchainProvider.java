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
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import java.util.List;

import javax.annotation.Nullable;

/**
 * Information about the JDK used by the <code>java_*</code> rules.
 */
@Immutable
public final class JavaToolchainProvider implements TransitiveInfoProvider {

  /** Returns the Java Toolchain associated with the rule being analyzed or {@code null}. */
  public static JavaToolchainProvider fromRuleContext(RuleContext ruleContext) {
    return ruleContext.getPrerequisite(":java_toolchain", Mode.TARGET, JavaToolchainProvider.class);
  }

  private final String sourceVersion;
  private final String targetVersion;
  @Nullable private final NestedSet<Artifact> bootclasspath;
  @Nullable private final NestedSet<Artifact> extclasspath;
  private final String encoding;
  private final ImmutableList<String> javacOptions;
  private final ImmutableList<String> javacJvmOptions;
  @Nullable private final Artifact javac;
  @Nullable private final Artifact javaBuilder;
  @Nullable private final Artifact headerCompiler;
  @Nullable private final Artifact singleJar;
  @Nullable private final Artifact genClass;
  @Nullable private final FilesToRunProvider ijar;

  public JavaToolchainProvider(
      JavaToolchainData data,
      @Nullable NestedSet<Artifact> bootclasspath,
      @Nullable NestedSet<Artifact> extclasspath,
      List<String> defaultJavacFlags,
      List<String> defaultJavacJvmOpts,
      @Nullable Artifact javac,
      @Nullable Artifact javaBuilder,
      @Nullable Artifact headerCompiler,
      @Nullable Artifact singleJar,
      @Nullable Artifact genClass,
      @Nullable FilesToRunProvider ijar) {
    this.sourceVersion = checkNotNull(data.getSourceVersion(), "sourceVersion must not be null");
    this.targetVersion = checkNotNull(data.getTargetVersion(), "targetVersion must not be null");
    this.bootclasspath = bootclasspath;
    this.extclasspath = extclasspath;
    this.encoding = checkNotNull(data.getEncoding(), "encoding must not be null");
    this.javac = javac;
    this.javaBuilder = javaBuilder;
    this.headerCompiler = headerCompiler;
    this.singleJar = singleJar;
    this.genClass = genClass;
    this.ijar = ijar;

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

  /** Returns the {@link Artifact} of the javac jar */
  @Nullable
  public Artifact getJavac() {
    return javac;
  }

  /** Returns the {@link Artifact} of the JavaBuilder deploy jar */
  @Nullable
  public Artifact getJavaBuilder() {
    return javaBuilder;
  }

  /** @return the {@link Artifact} of the Header Compiler deploy jar */
  @Nullable
  public Artifact getHeaderCompiler() {
    return headerCompiler;
  }

  /** Returns the {@link Artifact} of the SingleJar deploy jar */
  @Nullable
  public Artifact getSingleJar() {
    return singleJar;
  }

  /** Returns the {@link Artifact} of the GenClass deploy jar */
  @Nullable
  public Artifact getGenClass() {
    return genClass;
  }

  /** Returns the ijar executable */
  @Nullable
  public FilesToRunProvider getIjar() {
    return ijar;
  }
}
