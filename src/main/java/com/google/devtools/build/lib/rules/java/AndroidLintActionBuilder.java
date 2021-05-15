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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import javax.annotation.Nullable;

/** Helper to create Android Lint actions. */
class AndroidLintActionBuilder {
  private AndroidLintActionBuilder() {}

  private static final ParamFileInfo PARAM_FILE_INFO =
      ParamFileInfo.builder(ParameterFileType.UNQUOTED).setCharset(UTF_8).build();

  /** Creates and registers Android Lint action if needed and returns action's output if created. */
  @Nullable
  static Artifact create(
      RuleContext ruleContext,
      JavaConfiguration config,
      JavaTargetAttributes attributes,
      BootClassPathInfo bootClassPathInfo,
      JavaCommon common,
      JavaCompileOutputs<Artifact> outputs) {
    if (!config.runAndroidLint()
        || !attributes.hasSources()
        || JavaCommon.isNeverLink(ruleContext)) {
      // Note Javac doesn't run when depending on neverlink library, so we also skip Android Lint.
      return null;
    }
    if (config.limitAndroidLintToAndroidCompatible()
        && !JavaCommon.getConstraints(ruleContext).contains("android")) {
      return null;
    }
    JavaToolchainProvider toolchain = JavaToolchainProvider.from(ruleContext);
    AndroidLintTool androidLint = toolchain.getAndroidLint();
    if (androidLint == null) {
      ruleContext.ruleError(
          "android_lint_wrapper not set in java_toolchain: " + toolchain.getToolchainLabel());
      return null;
    }

    ImmutableList<Artifact> allSrcJars = attributes.getSourceJars();
    if (outputs.genSource() != null) {
      allSrcJars =
          ImmutableList.<Artifact>builder().addAll(allSrcJars).add(outputs.genSource()).build();
    }
    NestedSet<Artifact> classpath = attributes.getCompileTimeClassPath();
    if (!bootClassPathInfo.auxiliary().isEmpty()) {
      classpath =
          NestedSetBuilder.<Artifact>naiveLinkOrder()
              .addTransitive(bootClassPathInfo.auxiliary())
              .addTransitive(classpath)
              .build();
    }

    CustomCommandLine.Builder cmd = CustomCommandLine.builder();
    cmd.addExecPaths("--sources", attributes.getSourceFiles())
        .addExecPaths("--source_jars", allSrcJars)
        .addExecPaths("--bootclasspath", bootClassPathInfo.bootclasspath())
        .addExecPaths("--classpath", classpath)
        .addExecPaths("--plugins", attributes.plugins().plugins().processorClasspath())
        .addLabel("--target_label", ruleContext.getLabel());
    ImmutableList<String> javacopts =
        common.getJavacOpts().stream().map(StringCanonicalizer::intern).collect(toImmutableList());
    if (!javacopts.isEmpty()) {
      cmd.addAll("--javacopts", javacopts);
      // terminate --javacopts with `--` to support javac flags that start with `--`
      cmd.add("--");
    }
    cmd.add("--lintopts");
    cmd.addAll(androidLint.options());

    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    for (JavaPackageConfigurationProvider provider : androidLint.packageConfiguration()) {
      if (provider.matches(ruleContext.getLabel())) {
        cmd.addAll(provider.javacopts());
        inputs.addTransitive(provider.data());
      }
    }

    Artifact result =
        ruleContext.getPackageRelativeArtifact(
            ruleContext.getLabel().getName() + "_android_lint_output.xml",
            ruleContext.getBinOrGenfilesDirectory());
    cmd.addExecPath("--xml", result);

    SpawnAction.Builder spawnAction = new SpawnAction.Builder();
    androidLint.tool().buildCommandLine(spawnAction.executableArguments(), toolchain, inputs);
    ruleContext.registerAction(
        spawnAction
            .addCommandLine(cmd.build(), PARAM_FILE_INFO)
            .addInputs(attributes.getSourceFiles())
            .addInputs(allSrcJars)
            .addTransitiveInputs(bootClassPathInfo.bootclasspath())
            .addTransitiveInputs(classpath)
            .addTransitiveInputs(attributes.plugins().plugins().processorClasspath())
            .addTransitiveInputs(attributes.plugins().plugins().data())
            .addTransitiveInputs(inputs.build())
            .addOutput(result)
            .setMnemonic("AndroidLint")
            .setProgressMessage("Running Android Lint for: %s", ruleContext.getLabel())
            .build(ruleContext));
    return result;
  }
}
