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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ActionConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ArtifactNamePatternMapper;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Feature;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcToolchainConfigInfoApi;
import java.util.List;
import net.starlark.java.annot.StarlarkMethod;

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
  private final ImmutableList<StarlarkInfo> toolPaths;
  private final ImmutableList<StarlarkInfo> makeVariables;
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
      ImmutableList<StarlarkInfo> toolPaths,
      ImmutableList<StarlarkInfo> makeVariables,
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

  public ImmutableList<ActionConfig> getActionConfigs() {
    return actionConfigs;
  }

  public ImmutableList<Feature> getFeatures() {
    return features;
  }

  public ArtifactNamePatternMapper getArtifactNamePatterns() {
    return artifactNamePatterns;
  }

  @StarlarkMethod(name = "cxx_builtin_include_directories", documented = false, structField = true)
  public List<String> getCxxBuiltinIncludeDirectoriesForStarlark() {
    return getCxxBuiltinIncludeDirectories();
  }

  public ImmutableList<String> getCxxBuiltinIncludeDirectories() {
    return cxxBuiltinIncludeDirectories;
  }

  @StarlarkMethod(name = "toolchain_id", documented = false, structField = true)
  public String getToolchainIdentifierForStarlark() {
    return getToolchainIdentifier();
  }

  public String getToolchainIdentifier() {
    return toolchainIdentifier;
  }

  @StarlarkMethod(name = "target_system_name", documented = false, structField = true)
  public String getTargetSystemNameForStarlark() {
    return getTargetSystemName();
  }

  public String getTargetSystemName() {
    return targetSystemName;
  }

  @StarlarkMethod(name = "target_cpu", documented = false, structField = true)
  public String getTargetCpuForStarlark() {
    return getTargetCpu();
  }

  public String getTargetCpu() {
    return targetCpu;
  }

  @StarlarkMethod(name = "target_libc", documented = false, structField = true)
  public String getTargetLibcForStarlark() {
    return getTargetLibc();
  }

  public String getTargetLibc() {
    return targetLibc;
  }

  @StarlarkMethod(name = "compiler", documented = false, structField = true)
  public String getCompilerForStarlark() {
    return getCompiler();
  }

  public String getCompiler() {
    return compiler;
  }

  @StarlarkMethod(name = "abi_version", documented = false, structField = true)
  public String getAbiVersionForStarlark() {
    return getAbiVersion();
  }

  public String getAbiVersion() {
    return abiVersion;
  }

  @StarlarkMethod(name = "abi_libc_version", documented = false, structField = true)
  public String getAbiLibcVersionForStarlark() {
    return getAbiLibcVersion();
  }

  public String getAbiLibcVersion() {
    return abiLibcVersion;
  }

  @StarlarkMethod(name = "tool_paths", documented = false, structField = true)
  public ImmutableList<StarlarkInfo> getToolPathsForStarlark() {
    return getToolPaths();
  }

  /** Returns a list of paths of the tools in the form Pair<toolName, path>. */
  public ImmutableList<StarlarkInfo> getToolPaths() {
    return toolPaths;
  }

  @StarlarkMethod(name = "make_variables", documented = false, structField = true)
  public ImmutableList<StarlarkInfo> getMakevariablesForStarlark() {
    return getMakeVariables();
  }

  /** Returns a list of make variables that have the form Pair<name, value>. */
  public ImmutableList<StarlarkInfo> getMakeVariables() {
    return makeVariables;
  }

  @StarlarkMethod(name = "builtin_sysroot", documented = false, structField = true)
  public String getBuiltinSysrootForStarlark() {
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
