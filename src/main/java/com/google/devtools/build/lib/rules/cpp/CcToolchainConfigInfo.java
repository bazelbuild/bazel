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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ActionConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ArtifactNamePatternMapper;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Feature;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcToolchainConfigInfoApi;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import java.util.List;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;

/** Information describing C++ toolchain derived from CROSSTOOL file. */
@Immutable
public class CcToolchainConfigInfo extends NativeInfo implements CcToolchainConfigInfoApi {

  /** Singleton provider instance for {@link CcToolchainConfigInfo}. */
  public static final Provider PROVIDER = new Provider();

  private final ImmutableList<ActionConfig> actionConfigs;
  private final ImmutableList<Feature> features;
  private final ArtifactNamePatternMapper artifactNamePatterns;
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

  CcToolchainConfigInfo(
      ImmutableList<ActionConfig> actionConfigs,
      ImmutableList<Feature> features,
      ArtifactNamePatternMapper artifactNamePatterns,
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
      String builtinSysroot) {
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
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  @VisibleForTesting // Only called by tests.
  public static CcToolchainConfigInfo fromToolchainForTestingOnly(CToolchain toolchain)
      throws EvalException {
    ImmutableList.Builder<ActionConfig> actionConfigBuilder = ImmutableList.builder();
    for (CToolchain.ActionConfig actionConfig : toolchain.getActionConfigList()) {
      actionConfigBuilder.add(new ActionConfig(actionConfig));
    }

    ImmutableList.Builder<Feature> featureBuilder = ImmutableList.builder();
    for (CToolchain.Feature feature : toolchain.getFeatureList()) {
      featureBuilder.add(new Feature(feature));
    }

    ArtifactNamePatternMapper.Builder artifactNamePatternBuilder =
        new ArtifactNamePatternMapper.Builder();
    for (CToolchain.ArtifactNamePattern artifactNamePattern :
        toolchain.getArtifactNamePatternList()) {
      ArtifactCategory foundCategory = null;
      for (ArtifactCategory artifactCategory : ArtifactCategory.values()) {
        if (artifactNamePattern.getCategoryName().equals(artifactCategory.getCategoryName())) {
          foundCategory = artifactCategory;
          break;
        }
      }
      Preconditions.checkNotNull(foundCategory, artifactNamePattern);
      String extension = artifactNamePattern.getExtension();
      Preconditions.checkState(
          foundCategory.getAllowedExtensions().contains(extension),
          "%s had extension not in %s",
          artifactNamePattern,
          foundCategory);
      artifactNamePatternBuilder.addOverride(
          foundCategory, artifactNamePattern.getPrefix(), extension);
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
        toolchain.getBuiltinSysroot());
  }

  public ImmutableList<ActionConfig> getActionConfigs() {
    return actionConfigs;
  }

  public ImmutableList<Feature> getFeatures() {
    return features;
  }

  public ArtifactNamePatternMapper getArtifactNamePatterns() {
    return artifactNamePatterns;
  }

  @StarlarkMethod(
      name = "cxx_builtin_include_directories",
      documented = false,
      useStarlarkThread = true)
  public List<String> getCxxBuiltinIncludeDirectoriesForStarlark(StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getCxxBuiltinIncludeDirectories();
  }

  public ImmutableList<String> getCxxBuiltinIncludeDirectories() {
    return cxxBuiltinIncludeDirectories;
  }

  @StarlarkMethod(name = "toolchain_id", documented = false, useStarlarkThread = true)
  public String getToolchainIdentifierForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getToolchainIdentifier();
  }

  public String getToolchainIdentifier() {
    return toolchainIdentifier;
  }

  @StarlarkMethod(name = "target_system_name", documented = false, useStarlarkThread = true)
  public String getTargetSystemNameForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getTargetSystemName();
  }

  public String getTargetSystemName() {
    return targetSystemName;
  }

  @StarlarkMethod(name = "target_cpu", documented = false, useStarlarkThread = true)
  public String getTargetCpuForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getTargetCpu();
  }

  public String getTargetCpu() {
    return targetCpu;
  }

  @StarlarkMethod(name = "target_libc", documented = false, useStarlarkThread = true)
  public String getTargetLibcForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getTargetLibc();
  }

  public String getTargetLibc() {
    return targetLibc;
  }

  @StarlarkMethod(name = "compiler", documented = false, useStarlarkThread = true)
  public String getCompilerForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getCompiler();
  }

  public String getCompiler() {
    return compiler;
  }

  @StarlarkMethod(name = "abi_version", documented = false, useStarlarkThread = true)
  public String getAbiVersionForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getAbiVersion();
  }

  public String getAbiVersion() {
    return abiVersion;
  }

  @StarlarkMethod(name = "abi_libc_version", documented = false, useStarlarkThread = true)
  public String getAbiLibcVersionForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getAbiLibcVersion();
  }

  public String getAbiLibcVersion() {
    return abiLibcVersion;
  }

  @StarlarkMethod(name = "tool_paths", documented = false, useStarlarkThread = true)
  public ImmutableList<Tuple> getToolPathsForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getToolPaths().stream()
        .map(p -> Tuple.of(p.getFirst(), p.getSecond()))
        .collect(toImmutableList());
  }

  /** Returns a list of paths of the tools in the form Pair<toolName, path>. */
  public ImmutableList<Pair<String, String>> getToolPaths() {
    return toolPaths;
  }

  @StarlarkMethod(name = "make_variables", documented = false, useStarlarkThread = true)
  public ImmutableList<Tuple> getMakevariablesForStarlark(StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getMakeVariables().stream()
        .map(p -> Tuple.of(p.getFirst(), p.getSecond()))
        .collect(toImmutableList());
  }

  /** Returns a list of make variables that have the form Pair<name, value>. */
  public ImmutableList<Pair<String, String>> getMakeVariables() {
    return makeVariables;
  }

  @StarlarkMethod(name = "builtin_sysroot", documented = false, useStarlarkThread = true)
  public String getBuiltinSysrootForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getBuiltinSysroot();
  }

  public String getBuiltinSysroot() {
    return builtinSysroot;
  }

  /** Provider class for {@link CcToolchainConfigInfo} objects. */
  public static class Provider extends BuiltinProvider<CcToolchainConfigInfo>
      implements CcToolchainConfigInfoApi.Provider {

    private Provider() {
      super("CcToolchainConfigInfo", CcToolchainConfigInfo.class);
    }
  }
}
