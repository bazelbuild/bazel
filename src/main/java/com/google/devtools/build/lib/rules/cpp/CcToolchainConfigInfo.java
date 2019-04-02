// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;


import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ActionConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ArtifactNamePattern;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.EnvEntry;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.EnvSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Feature;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FlagGroup;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FlagSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Tool;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.VariableWithValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.WithFeatureSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.Expandable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcToolchainConfigInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;

/** Information describing C++ toolchain derived from CROSSTOOL file. */
@Immutable
public class CcToolchainConfigInfo extends NativeInfo implements CcToolchainConfigInfoApi {
  public static final NativeProvider<CcToolchainConfigInfo> PROVIDER =
      new NativeProvider<CcToolchainConfigInfo>(
          CcToolchainConfigInfo.class, "CcToolchainConfigInfo") {};

  private final ImmutableList<ActionConfig> actionConfigs;
  private final ImmutableList<Feature> features;
  private final ImmutableList<ArtifactNamePattern> artifactNamePatterns;
  private final ImmutableList<String> cxxBuiltinIncludeDirectories;

  private final String toolchainIdentifier;
  private final String hostSystemName;
  private final String targetSystemName;
  private final String targetCpu;
  private final String targetLibc;
  private final String compiler;
  private final String abiVersion;
  private final String abiLibcVersion;
  private final ImmutableList<Pair<String, String>> toolPaths;
  private final ImmutableList<Pair<String, String>> makeVariables;
  private final String builtinSysroot;
  private final String ccTargetOs;
  private final String proto;

  @AutoCodec.Instantiator
  public CcToolchainConfigInfo(
      ImmutableList<ActionConfig> actionConfigs,
      ImmutableList<Feature> features,
      ImmutableList<ArtifactNamePattern> artifactNamePatterns,
      ImmutableList<String> cxxBuiltinIncludeDirectories,
      String toolchainIdentifier,
      String hostSystemName,
      String targetSystemName,
      String targetCpu,
      String targetLibc,
      String compiler,
      String abiVersion,
      String abiLibcVersion,
      ImmutableList<Pair<String, String>> toolPaths,
      ImmutableList<Pair<String, String>> makeVariables,
      String builtinSysroot,
      String ccTargetOs,
      String proto) {
    super(PROVIDER);
    this.actionConfigs = actionConfigs;
    this.features = features;
    this.artifactNamePatterns = artifactNamePatterns;
    this.cxxBuiltinIncludeDirectories = cxxBuiltinIncludeDirectories;
    this.toolchainIdentifier = toolchainIdentifier;
    this.hostSystemName = hostSystemName;
    this.targetSystemName = targetSystemName;
    this.targetCpu = targetCpu;
    this.targetLibc = targetLibc;
    this.compiler = compiler;
    this.abiVersion = abiVersion;
    this.abiLibcVersion = abiLibcVersion;
    this.toolPaths = toolPaths;
    this.makeVariables = makeVariables;
    this.builtinSysroot = builtinSysroot;
    this.ccTargetOs = ccTargetOs;
    this.proto = proto;
  }

  public static CcToolchainConfigInfo fromToolchain(CToolchain toolchain) throws EvalException {
    ImmutableList.Builder<ActionConfig> actionConfigBuilder = ImmutableList.builder();
    for (CToolchain.ActionConfig actionConfig : toolchain.getActionConfigList()) {
      actionConfigBuilder.add(new ActionConfig(actionConfig));
    }

    ImmutableList.Builder<Feature> featureBuilder = ImmutableList.builder();
    for (CToolchain.Feature feature : toolchain.getFeatureList()) {
      featureBuilder.add(new Feature(feature));
    }

    ImmutableList.Builder<ArtifactNamePattern> artifactNamePatternBuilder = ImmutableList.builder();
    for (CToolchain.ArtifactNamePattern artifactNamePattern :
        toolchain.getArtifactNamePatternList()) {
      artifactNamePatternBuilder.add(new ArtifactNamePattern(artifactNamePattern));
    }

    return new CcToolchainConfigInfo(
        actionConfigBuilder.build(),
        featureBuilder.build(),
        artifactNamePatternBuilder.build(),
        ImmutableList.copyOf(toolchain.getCxxBuiltinIncludeDirectoryList()),
        toolchain.getToolchainIdentifier(),
        toolchain.getHostSystemName(),
        toolchain.getTargetSystemName(),
        toolchain.getTargetCpu(),
        toolchain.getTargetLibc(),
        toolchain.getCompiler(),
        toolchain.getAbiVersion(),
        toolchain.getAbiLibcVersion(),
        toolchain.getToolPathList().stream()
            .map(a -> Pair.of(a.getName(), a.getPath()))
            .collect(ImmutableList.toImmutableList()),
        toolchain.getMakeVariableList().stream()
            .map(makeVariable -> Pair.of(makeVariable.getName(), makeVariable.getValue()))
            .collect(ImmutableList.toImmutableList()),
        toolchain.getBuiltinSysroot(),
        toolchain.getCcTargetOs(),
        /* proto= */ "");
  }

  public ImmutableList<ActionConfig> getActionConfigs() {
    return actionConfigs;
  }

  public ImmutableList<Feature> getFeatures() {
    return features;
  }

  public ImmutableList<ArtifactNamePattern> getArtifactNamePatterns() {
    return artifactNamePatterns;
  }

  public ImmutableList<String> getCxxBuiltinIncludeDirectories() {
    return cxxBuiltinIncludeDirectories;
  }

  public String getToolchainIdentifier() {
    return toolchainIdentifier;
  }

  public String getHostSystemName() {
    return hostSystemName;
  }

  public String getTargetSystemName() {
    return targetSystemName;
  }

  public String getTargetCpu() {
    return targetCpu;
  }

  public String getTargetLibc() {
    return targetLibc;
  }

  public String getCompiler() {
    return compiler;
  }

  public String getAbiVersion() {
    return abiVersion;
  }

  public String getAbiLibcVersion() {
    return abiLibcVersion;
  }

  /** Returns a list of paths of the tools in the form Pair<toolName, path>. */
  public ImmutableList<Pair<String, String>> getToolPaths() {
    return toolPaths;
  }

  /** Returns a list of make variables that have the form Pair<name, value>. */
  public ImmutableList<Pair<String, String>> getMakeVariables() {
    return makeVariables;
  }

  public String getBuiltinSysroot() {
    return builtinSysroot;
  }

  public String getCcTargetOs() {
    return ccTargetOs;
  }

  @Override
  public String getProto() {
    return proto;
  }

  private static CToolchain.WithFeatureSet withFeatureSetToProto(WithFeatureSet withFeatureSet) {
    return CToolchain.WithFeatureSet.newBuilder()
        .addAllFeature(withFeatureSet.getFeatures())
        .addAllNotFeature(withFeatureSet.getNotFeatures())
        .build();
  }

  private static CToolchain.EnvEntry envEntryToProto(EnvEntry envEntry) {
    return CToolchain.EnvEntry.newBuilder()
        .setKey(envEntry.getKey())
        .setValue(envEntry.getValue())
        .build();
  }

  private static CToolchain.EnvSet envSetToProto(EnvSet envSet) {
    return CToolchain.EnvSet.newBuilder()
        .addAllAction(envSet.getActions())
        .addAllEnvEntry(
            envSet.getEnvEntries().stream()
                .map(envEntry -> envEntryToProto(envEntry))
                .collect(ImmutableList.toImmutableList()))
        .addAllWithFeature(
            envSet.getWithFeatureSets().stream()
                .map(withFeatureSet -> withFeatureSetToProto(withFeatureSet))
                .collect(ImmutableList.toImmutableList()))
        .build();
  }

  private static CToolchain.FlagGroup flagGroupToProto(FlagGroup flagGroup) {
    ImmutableList.Builder<CToolchain.FlagGroup> flagGroups = ImmutableList.builder();
    ImmutableList.Builder<String> flags = ImmutableList.builder();
    for (Expandable expandable : flagGroup.getExpandables()) {
      if (expandable instanceof FlagGroup) {
        flagGroups.add(flagGroupToProto((FlagGroup) expandable));
      } else {
        flags.add(((CcToolchainFeatures.Flag) expandable).getString());
      }
    }

    CToolchain.FlagGroup.Builder flagGroupBuilder = CToolchain.FlagGroup.newBuilder();
    if (flagGroup.getIterateOverVariable() != null) {
      flagGroupBuilder.setIterateOver(flagGroup.getIterateOverVariable());
    }
    if (flagGroup.getExpandIfTrue() != null) {
      flagGroupBuilder.setExpandIfTrue(flagGroup.getExpandIfTrue());
    }
    if (flagGroup.getExpandIfFalse() != null) {
      flagGroupBuilder.setExpandIfFalse(flagGroup.getExpandIfFalse());
    }
    if (flagGroup.getExpandIfEqual() != null) {
      flagGroupBuilder.setExpandIfEqual(variableWithValueFromProto(flagGroup.getExpandIfEqual()));
    }
    return flagGroupBuilder
        .addAllExpandIfAllAvailable(flagGroup.getExpandIfAllAvailable())
        .addAllExpandIfNoneAvailable(flagGroup.getExpandIfNoneAvailable())
        .addAllFlagGroup(flagGroups.build())
        .addAllFlag(flags.build())
        .build();
  }

  private static CToolchain.VariableWithValue variableWithValueFromProto(
      VariableWithValue variable) {
    return CToolchain.VariableWithValue.newBuilder()
        .setValue(variable.getValue())
        .setVariable(variable.getVariable())
        .build();
  }

  static CToolchain.Feature featureToProto(Feature feature) {
    return CToolchain.Feature.newBuilder()
        .setName(feature.getName())
        .setEnabled(feature.isEnabled())
        .addAllFlagSet(
            feature.getFlagSets().stream()
                .map(flagSet -> flagSetToProto(flagSet, /* forActionConfig= */ false))
                .collect(ImmutableList.toImmutableList()))
        .addAllEnvSet(
            feature.getEnvSets().stream()
                .map(envSet -> envSetToProto(envSet))
                .collect(ImmutableList.toImmutableList()))
        .addAllRequires(
            feature.getRequires().stream()
                .map(featureSet -> featureSetToProto(featureSet))
                .collect(ImmutableList.toImmutableList()))
        .addAllImplies(feature.getImplies())
        .addAllProvides(feature.getProvides())
        .build();
  }

  private static CToolchain.FlagSet flagSetToProto(FlagSet flagSet, boolean forActionConfig) {
    CToolchain.FlagSet.Builder flagSetBuilder =
        CToolchain.FlagSet.newBuilder()
            .addAllFlagGroup(
                flagSet.getFlagGroups().stream()
                    .map(flagGroup -> flagGroupToProto(flagGroup))
                    .collect(ImmutableList.toImmutableList()))
            .addAllWithFeature(
                flagSet.getWithFeatureSets().stream()
                    .map(withFeatureSet -> withFeatureSetToProto(withFeatureSet))
                    .collect(ImmutableList.toImmutableList()))
            .addAllExpandIfAllAvailable(flagSet.getExpandIfAllAvailable());
    if (!forActionConfig) {
      flagSetBuilder.addAllAction(flagSet.getActions());
    }
    return flagSetBuilder.build();
  }

  private static CToolchain.FeatureSet featureSetToProto(ImmutableSet<String> features) {
    return CToolchain.FeatureSet.newBuilder().addAllFeature(features).build();
  }

  private static CToolchain.Tool toolToProto(Tool tool) {
    return CToolchain.Tool.newBuilder()
        .setToolPath(tool.getToolPathFragment().toString())
        .addAllWithFeature(
            tool.getWithFeatureSetSets().stream()
                .map(withFeatureSet -> withFeatureSetToProto(withFeatureSet))
                .collect(ImmutableList.toImmutableList()))
        .addAllExecutionRequirement(tool.getExecutionRequirements())
        .build();
  }

  static CToolchain.ActionConfig actionConfigToProto(ActionConfig actionConfig) {
    return CToolchain.ActionConfig.newBuilder()
        .setConfigName(actionConfig.getName())
        .setActionName(actionConfig.getActionName())
        .setEnabled(actionConfig.isEnabled())
        .addAllTool(
            actionConfig.getTools().stream()
                .map(tool -> toolToProto(tool))
                .collect(ImmutableList.toImmutableList()))
        .addAllFlagSet(
            actionConfig.getFlagSets().stream()
                .map(flagSet -> flagSetToProto(flagSet, /* forActionConfig= */ true))
                .collect(ImmutableList.toImmutableList()))
        .addAllImplies(actionConfig.getImplies())
        .build();
  }
}
