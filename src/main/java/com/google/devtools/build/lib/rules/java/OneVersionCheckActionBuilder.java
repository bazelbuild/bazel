// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** Utility for generating a call to the one-version binary. */
public final class OneVersionCheckActionBuilder {

  private OneVersionCheckActionBuilder() {}

  private OneVersionEnforcementLevel enforcementLevel;
  private Artifact outputArtifact;
  private JavaToolchainProvider javaToolchain;
  private NestedSet<Artifact> jarsToCheck;

  public static OneVersionCheckActionBuilder newBuilder() {
    return new OneVersionCheckActionBuilder();
  }

  public OneVersionCheckActionBuilder useToolchain(JavaToolchainProvider toolchain) {
    javaToolchain = toolchain;
    return this;
  }

  public OneVersionCheckActionBuilder checkJars(NestedSet<Artifact> jarsToCheck) {
    this.jarsToCheck = jarsToCheck;
    return this;
  }

  public OneVersionCheckActionBuilder outputArtifact(Artifact outputArtifact) {
    this.outputArtifact = outputArtifact;
    return this;
  }

  public OneVersionCheckActionBuilder withEnforcementLevel(
      OneVersionEnforcementLevel enforcementLevel) {
    Preconditions.checkArgument(
        enforcementLevel != OneVersionEnforcementLevel.OFF,
        "one version enforcement actions shouldn't be built if the enforcement "
            + "level is set to off");
    this.enforcementLevel = enforcementLevel;
    return this;
  }

  public Artifact build(RuleContext ruleContext) {
    Preconditions.checkNotNull(enforcementLevel);
    Preconditions.checkNotNull(outputArtifact);
    Preconditions.checkNotNull(javaToolchain);
    Preconditions.checkNotNull(jarsToCheck);

    Artifact oneVersionTool = javaToolchain.getOneVersionBinary();
    Artifact oneVersionAllowlist = javaToolchain.getOneVersionAllowlist();
    if (oneVersionTool == null || oneVersionAllowlist == null) {
      addRuleErrorForMissingArtifacts(ruleContext, javaToolchain);
      return outputArtifact;
    }

    CustomCommandLine.Builder oneVersionArgsBuilder =
        CustomCommandLine.builder()
            .addExecPath("--output", outputArtifact)
            .addExecPath("--whitelist", oneVersionAllowlist);
    if (enforcementLevel == OneVersionEnforcementLevel.WARNING) {
      oneVersionArgsBuilder.add("--succeed_on_found_violations");
    }
    oneVersionArgsBuilder.addAll("--inputs", jarAndTargetVectorArg(jarsToCheck));
    CustomCommandLine oneVersionArgs = oneVersionArgsBuilder.build();
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addOutput(outputArtifact)
            .addInput(oneVersionAllowlist)
            .addTransitiveInputs(jarsToCheck)
            .setExecutable(oneVersionTool)
            .addCommandLine(
                oneVersionArgs,
                ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED).setUseAlways(true).build())
            .setMnemonic("JavaOneVersion")
            .setProgressMessage("Checking for one-version violations in %s", ruleContext.getLabel())
            .build(ruleContext));
    return outputArtifact;
  }

  public static void addRuleErrorForMissingArtifacts(
      RuleContext ruleContext, JavaToolchainProvider javaToolchain) {
    ruleContext.ruleError(
        String.format(
            "one version enforcement was requested but it is not supported by the current "
                + "Java toolchain '%s'; see the "
                + "java_toolchain.oneversion and java_toolchain.oneversion_allowlist "
                + "attributes",
            javaToolchain.getToolchainLabel()));
  }

  static VectorArg<String> jarAndTargetVectorArg(NestedSet<Artifact> jarsToCheck) {
    return VectorArg.of(jarsToCheck).mapped(EXPAND_TO_JAR_AND_TARGET);
  }

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final CommandLineItem.MapFn<Artifact> EXPAND_TO_JAR_AND_TARGET =
      (jar, args) ->
          args.accept(jar.getExecPathString() + "," + getArtifactOwnerGeneralizedLabel(jar));

  private static String getArtifactOwnerGeneralizedLabel(Artifact artifact) {
    Label label = checkNotNull(artifact.getOwnerLabel(), artifact);
    return label.getRepository().isDefault() || label.getRepository().isMain()
        ? label.toString()
        // Escape '@' prefix for .params file.
        : "@" + label;
  }
}
