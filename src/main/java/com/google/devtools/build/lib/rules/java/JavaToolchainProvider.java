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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Information about the JDK used by the <code>java_*</code> rules.
 */
@AutoValue
@Immutable
public abstract class JavaToolchainProvider implements TransitiveInfoProvider {

  /** Returns the Java Toolchain associated with the rule being analyzed or {@code null}. */
  public static JavaToolchainProvider fromRuleContext(RuleContext ruleContext) {
    return ruleContext.getPrerequisite(":java_toolchain", Mode.TARGET, JavaToolchainProvider.class);
  }

  public static JavaToolchainProvider create(
      Label label,
      JavaToolchainData data,
      NestedSet<Artifact> bootclasspath,
      NestedSet<Artifact> extclasspath,
      List<String> defaultJavacFlags,
      Artifact javac,
      Artifact javaBuilder,
      @Nullable Artifact headerCompiler,
      boolean forciblyDisableHeaderCompilation,
      Artifact singleJar,
      Artifact oneVersion,
      Artifact oneVersionWhitelist,
      Artifact genClass,
      @Nullable Artifact resourceJarBuilder,
      @Nullable Artifact timezoneData,
      FilesToRunProvider ijar,
      ImmutableListMultimap<String, String> compatibleJavacOptions) {
    return new AutoValue_JavaToolchainProvider(
        label,
        data.getSourceVersion(),
        data.getTargetVersion(),
        bootclasspath,
        extclasspath,
        data.getEncoding(),
        javac,
        javaBuilder,
        headerCompiler,
        forciblyDisableHeaderCompilation,
        singleJar,
        oneVersion,
        oneVersionWhitelist,
        genClass,
        resourceJarBuilder,
        timezoneData,
        ijar,
        compatibleJavacOptions,
        // merges the defaultJavacFlags from
        // {@link JavaConfiguration} with the flags from the {@code java_toolchain} rule.
        ImmutableList.<String>builder()
            .addAll(data.getJavacOptions())
            .addAll(defaultJavacFlags)
            .build(),
        data.getJvmOptions(),
        data.getJavacSupportsWorkers());
  }

  /** Returns the label for this {@code java_toolchain}. */
  public abstract Label getToolchainLabel();

  /** @return the input Java language level */
  public abstract String getSourceVersion();

  /** @return the target Java language level */
  public abstract String getTargetVersion();

  /** @return the target Java bootclasspath */
  public abstract NestedSet<Artifact> getBootclasspath();

  /** @return the target Java extclasspath */
  public abstract NestedSet<Artifact> getExtclasspath();

  /** @return the encoding for Java source files */
  public abstract String getEncoding();

  /** Returns the {@link Artifact} of the javac jar */
  public abstract Artifact getJavac();

  /** Returns the {@link Artifact} of the JavaBuilder deploy jar */
  public abstract Artifact getJavaBuilder();

  /** @return the {@link Artifact} of the Header Compiler deploy jar */
  @Nullable public abstract Artifact getHeaderCompiler();

  /**
   * Returns true if header compilation should be forcibly disabled, overriding
   * --java_header_compilation.
   */
  public abstract boolean getForciblyDisableHeaderCompilation();

  /** Returns the {@link Artifact} of the SingleJar deploy jar */
  public abstract Artifact getSingleJar();

  /**
   * Return the {@link Artifact} of the binary that enforces one-version compliance of java
   * binaries.
   */
  @Nullable
  public abstract Artifact getOneVersionBinary();

  /** Return the {@link Artifact} of the whitelist used by the one-version compliance checker. */
  @Nullable
  public abstract Artifact getOneVersionWhitelist();

  /** Returns the {@link Artifact} of the GenClass deploy jar */
  public abstract Artifact getGenClass();

  @Nullable
  public abstract Artifact getResourceJarBuilder();

  /**
   * Returns the {@link Artifact} of the latest timezone data resource jar that can be loaded by
   * Java 8 binaries.
   */
  @Nullable
  public abstract Artifact getTimezoneData();

  /** Returns the ijar executable */
  public abstract FilesToRunProvider getIjar();

  abstract ImmutableListMultimap<String, String> getCompatibleJavacOptions();

  /** @return the map of target environment-specific javacopts. */
  public ImmutableList<String> getCompatibleJavacOptions(String key) {
    return getCompatibleJavacOptions().get(key);
  }

  /** @return the list of default options for the java compiler */
  public abstract ImmutableList<String> getJavacOptions();

  /**
   * @return the list of default options for the JVM running the java compiler and associated tools.
   */
  public abstract ImmutableList<String> getJvmOptions();

  /** @return whether JavaBuilders supports running as a persistent worker or not */
  public abstract boolean getJavacSupportsWorkers();

  JavaToolchainProvider() {}
}
