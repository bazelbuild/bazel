// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Type;

import java.util.List;

/**
 * Information about the JDK used by the <code>java_*</code> rules.
 */
@Immutable
public final class JavaToolchainProvider implements TransitiveInfoProvider {

  private final ImmutableList<String> javacOptions;
  private final ImmutableList<String> javacJvmOptions;

  public JavaToolchainProvider(JavaToolchainData data, List<String> defaultJavacFlags,
      List<String> defaultJavacJvmOpts) {
    super();
    // merges the defaultJavacFlags from
    // {@link JavaConfiguration} with the flags from the {@code java_toolchain} rule.
    this.javacOptions = ImmutableList.<String>builder()
        .addAll(data.getJavacOptions())
        .addAll(defaultJavacFlags)
        .build();
    // merges the defaultJavaBuilderJvmFlags from
    // {@link JavaConfiguration} with the flags from the {@code java_toolchain} rule.
    this.javacJvmOptions = ImmutableList.<String>builder()
        .addAll(data.getJavacJvmOptions())
        .addAll(defaultJavacJvmOpts)
        .build();
  }

  /**
   * @return the list of default options for the java compiler
   */
  public ImmutableList<String> getJavacOptions() {
    return javacOptions;
  }

  /**
   * @return the list of default options for the JVM running the java compiler
   */
  public ImmutableList<String> getJavacJvmOptions() {
    return javacJvmOptions;
  }

  /**
   * An helper method to construct the list of javac options.
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
   * An helper method to construct the list of options to pass to the JVM running the java compiler.
   *
   * @param ruleContext The rule context of the current rule.
   * @return the list of flags provided by the {@code java_toolchain} rule merged with the one
   *         provided by the {@link JavaConfiguration} fragment.
   */
  public static List<String> getDefaultJavacJvmOptions(RuleContext ruleContext) {
    if (!ruleContext.getRule().isAttrDefined(":java_toolchain", Type.LABEL))  {
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
}
