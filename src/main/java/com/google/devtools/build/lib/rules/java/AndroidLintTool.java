// Copyright 2020 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkValue;

/** The Android Lint part of {@code java_toolchain}. */
@AutoValue
@AutoCodec
abstract class AndroidLintTool implements StarlarkValue {

  abstract JavaToolchainTool tool();

  abstract ImmutableList<String> options();

  abstract ImmutableList<JavaPackageConfigurationProvider> packageConfiguration();

  @Nullable
  static AndroidLintTool fromRuleContext(RuleContext ruleContext) {
    JavaToolchainTool tool =
        JavaToolchainTool.fromRuleContext(
            ruleContext, "android_lint_runner", "android_lint_data", "android_lint_jvm_opts");
    if (tool == null) {
      return null;
    }
    ImmutableList<String> options =
        ImmutableList.copyOf(ruleContext.attributes().get("android_lint_opts", Type.STRING_LIST));
    ImmutableList<JavaPackageConfigurationProvider> packageConfiguration =
        ImmutableList.copyOf(
            ruleContext.getPrerequisites(
                "android_lint_package_configuration", JavaPackageConfigurationProvider.class));
    return create(tool, options, packageConfiguration);
  }

  @AutoCodec.Instantiator
  static AndroidLintTool create(
      JavaToolchainTool tool,
      ImmutableList<String> options,
      ImmutableList<JavaPackageConfigurationProvider> packageConfiguration) {
    return new AutoValue_AndroidLintTool(tool, options, packageConfiguration);
  }

  @StarlarkMethod(name = "tool", documented = false, structField = true)
  public FilesToRunProvider starlarkTool() {
    return tool().tool();
  }

  @StarlarkMethod(name = "jvm_opts", documented = false, structField = true)
  public Sequence<String> starlarkJvmOpts() {
    return StarlarkList.immutableCopyOf(tool().jvmOpts());
  }

  @StarlarkMethod(name = "data", documented = false, structField = true)
  public Depset starlarkData() {
    return Depset.of(Artifact.TYPE, tool().data());
  }

  @StarlarkMethod(name = "lint_opts", documented = false, structField = true)
  public Sequence<String> starlarkLintOpts() {
    return StarlarkList.immutableCopyOf(options());
  }

  @StarlarkMethod(name = "package_config", documented = false, structField = true)
  public Sequence<?> starlarkPackageConfig() {
    return StarlarkList.immutableCopyOf(packageConfiguration());
  }
}
