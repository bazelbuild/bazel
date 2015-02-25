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

import java.util.List;

/**
 * Information about the JDK used by the <code>java_*</code> rules.
 */
@Immutable
public final class JavaToolchainProvider implements TransitiveInfoProvider {

  private final ImmutableList<String> javacOptions;

  public JavaToolchainProvider(String source, String target, String encoding,
      ImmutableList<String> xlint, ImmutableList<String> misc, List<String> defaultJavacFlags) {
    super();
    // merges the defaultJavacFlags from
    // {@link JavaConfiguration} with the flags from the {@code java_toolchain} rule.
    JavaToolchainData data = new JavaToolchainData(source, target, encoding, xlint, misc);
    this.javacOptions = ImmutableList.<String>builder()
        .addAll(data.getJavacOptions())
        .addAll(defaultJavacFlags)
        .build();
  }

  /**
   * @return the list of default options for the java compiler
   */
  public ImmutableList<String> getJavacOptions() {
    return javacOptions;
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
      ruleContext.ruleError("No java_toolchain implicit dependency found. This is probably because"
          + " your java configuration is not up-to-date.");
      return ImmutableList.of();
    }
    return javaToolchain.getJavacOptions();
  }
}
