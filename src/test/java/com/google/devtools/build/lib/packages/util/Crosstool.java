// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.util;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A helper class to create a crosstool package containing a CROSSTOOL file, and the various rules
 * needed for a mock - use this only for configured target tests, not for execution tests.
 */
public final class Crosstool {
  private static final ImmutableList<String> CROSSTOOL_BINARIES =
      ImmutableList.of("ar", "as", "compile", "dwp", "link", "objcopy", "llvm-profdata");

  /**
   * A class that contains relevant fields from either the CROSSTOOL file or the Starlark rule
   * implementation that are needed in order to generate the BUILD file.
   */
  public static final class CcToolchainConfig {
    private final String cpu;
    private final String compiler;
    private final String toolchainIdentifier;
    private final String hostSystemName;
    private final String targetSystemName;
    private final String abiVersion;
    private final String abiLibcVersion;
    private final String targetLibc;
    private final ImmutableList<String> features;
    private final ImmutableList<String> actionConfigs;
    private final ImmutableList<String> artifactNamePatterns;
    private final ImmutableList<Pair<String, String>> toolPaths;

    private CcToolchainConfig(
        String cpu,
        String compiler,
        String toolchainIdentifier,
        String hostSystemName,
        String targetSystemName,
        String abiVersion,
        String abiLibcVersion,
        String targetLibc,
        ImmutableList<String> features,
        ImmutableList<String> actionConfigs,
        ImmutableList<String> artifactNamePatterns,
        ImmutableList<Pair<String, String>> toolPaths) {
      this.cpu = cpu;
      this.compiler = compiler;
      this.toolchainIdentifier = toolchainIdentifier;
      this.hostSystemName = hostSystemName;
      this.targetSystemName = targetSystemName;
      this.abiVersion = abiVersion;
      this.abiLibcVersion = abiLibcVersion;
      this.targetLibc = targetLibc;
      this.features = features;
      this.actionConfigs = actionConfigs;
      this.artifactNamePatterns = artifactNamePatterns;
      this.toolPaths = toolPaths;
    }

    public static Builder builder() {
      return new Builder();
    }

    /** A Builder for {@link CcToolchainConfig}. */
    public static class Builder {
      private ImmutableList<String> features = ImmutableList.of();
      private ImmutableList<String> actionConfigs = ImmutableList.of();
      private ImmutableList<String> artifactNamePatterns = ImmutableList.of();
      private ImmutableList<Pair<String, String>> toolPaths = ImmutableList.of();

      public Builder withFeatures(String... features) {
        this.features = ImmutableList.copyOf(features);
        return this;
      }

      public Builder withActionConfigs(String... actionConfigs) {
        this.actionConfigs = ImmutableList.copyOf(actionConfigs);
        return this;
      }

      public Builder withArtifactNamePatterns(String... artifactNamePatterns) {
        this.artifactNamePatterns = ImmutableList.copyOf(artifactNamePatterns);
        return this;
      }

      public Builder withToolPaths(Pair<String, String>... toolPaths) {
        this.toolPaths = ImmutableList.copyOf(toolPaths);
        return this;
      }

      public CcToolchainConfig build() {
        return new CcToolchainConfig(
            /* cpu= */ "k8",
            /* compiler= */ "compiler",
            /* toolchainIdentifier= */ "mock-llvm-toolchain-k8",
            /* hostSystemName= */ "local",
            /* targetSystemName= */ "local",
            /* abiVersion= */ "local",
            /* abiLibcVersion= */ "local",
            /* targetLibc= */ "local",
            features,
            actionConfigs,
            artifactNamePatterns,
            toolPaths);
      }
    }

    public String getToolchainIdentifier() {
      return toolchainIdentifier;
    }

    public String getTargetCpu() {
      return cpu;
    }

    public String getCompiler() {
      return compiler;
    }

    public boolean hasStaticLinkCppRuntimesFeature() {
      return features.contains(CppRuleClasses.STATIC_LINK_CPP_RUNTIMES);
    }

    public static CcToolchainConfig getCcToolchainConfigForCpu(String cpu) {
      return new CcToolchainConfig(
          /* cpu= */ cpu,
          /* compiler= */ "mock-compiler-for-" + cpu,
          /* toolchainIdentifier= */ "mock-llvm-toolchain-for-" + cpu,
          /* hostSystemName= */ "mock-system-name-for-" + cpu,
          /* targetSystemName= */ "mock-target-system-name-for-" + cpu,
          /* abiVersion= */ "mock-abi-version-for-" + cpu,
          /* abiLibcVersion= */ "mock-abi-libc-for-" + cpu,
          /* targetLibc= */ "mock-libc-for-" + cpu,
          /* features= */ ImmutableList.of(),
          /* actionConfigs= */ ImmutableList.of(),
          /* artifactNamePatterns= */ ImmutableList.of(),
          /* toolPaths= */ ImmutableList.of());
    }

    public static CcToolchainConfig getDefaultCcToolchainConfig() {
      return getCcToolchainConfigForCpu("k8");
    }

    String getCcToolchainConfigRule() {
      ImmutableList<String> featuresList =
          features.stream()
              .map(feature -> "'" + feature + "'")
              .collect(ImmutableList.toImmutableList());
      ImmutableList<String> actionConfigsList =
          actionConfigs.stream()
              .map(config -> "'" + config + "'")
              .collect(ImmutableList.toImmutableList());
      ImmutableList<String> patternsList =
          artifactNamePatterns.stream()
              .map(pattern -> "'" + pattern + "'")
              .collect(ImmutableList.toImmutableList());
      ImmutableList<String> toolPathsList =
          toolPaths.stream()
              .map(toolPath -> String.format("'%s': '%s'", toolPath.first, toolPath.second))
              .collect(ImmutableList.toImmutableList());

      return Joiner.on("\n")
          .join(
              "cc_toolchain_config(",
              "  name = '" + cpu + "-" + compiler + "_config',",
              "  cpu = '" + cpu + "',",
              "  compiler = '" + compiler + "',",
              "  host_system_name = '" + hostSystemName + "',",
              "  target_system_name = '" + targetSystemName + "',",
              "  target_libc = '" + targetLibc + "',",
              "  abi_version = '" + abiVersion + "',",
              "  abi_libc_version = '" + abiLibcVersion + "',",
              String.format("  feature_names = [%s],", Joiner.on(",\n    ").join(featuresList)),
              String.format(
                  "  action_configs = [%s],", Joiner.on(",\n    ").join(actionConfigsList)),
              String.format(
                  "  artifact_name_patterns = [%s],", Joiner.on(",\n    ").join(patternsList)),
              String.format("  tool_paths = {%s},", Joiner.on(",\n    ").join(toolPathsList)),
              "  )");
    }
  }

  private final MockToolsConfig config;

  private final String crosstoolTop;
  private String version;
  private String crosstoolFileContents;
  private ImmutableList<String> archs;
  private boolean supportsHeaderParsing;
  private ImmutableList<CcToolchainConfig> ccToolchainConfigList;
  private final boolean disableCrosstool;

  Crosstool(MockToolsConfig config, String crosstoolTop, boolean disableCrosstool) {
    this.config = config;
    this.crosstoolTop = crosstoolTop;
    this.disableCrosstool = disableCrosstool;
  }

  public Crosstool setCrosstoolFile(String version, String crosstoolFileContents) {
    this.version = version;
    this.crosstoolFileContents = crosstoolFileContents;
    return this;
  }

  public Crosstool setSupportedArchs(ImmutableList<String> archs) {
    this.archs = archs;
    return this;
  }

  public Crosstool setSupportsHeaderParsing(boolean supportsHeaderParsing) {
    this.supportsHeaderParsing = supportsHeaderParsing;
    return this;
  }

  public Crosstool setToolchainConfigs(ImmutableList<CcToolchainConfig> ccToolchainConfigs) {
    this.ccToolchainConfigList = ccToolchainConfigs;
    return this;
  }

  public void write() throws IOException {
    Set<String> runtimes = new HashSet<>();
    StringBuilder compilationTools = new StringBuilder();
    for (String compilationTool : CROSSTOOL_BINARIES) {
      Collection<String> archTargets = new ArrayList<>();
      for (String arch : archs) {
        archTargets.add(compilationTool + '-' + arch);
      }

      compilationTools.append(
          String.format(
              "filegroup(name = '%s', srcs = ['%s'])\n",
              compilationTool,
              Joiner.on("', '").join(archTargets)));
      for (String archTarget : archTargets) {
        compilationTools.append(
            String.format("filegroup(name = '%s', srcs = [':everything-multilib'])\n", archTarget));
      }
    }
    ImmutableList<CcToolchainConfig> ccToolchainConfigs;
    if (disableCrosstool) {
      ccToolchainConfigs = ccToolchainConfigList;
    } else {
      CrosstoolConfig.CrosstoolRelease.Builder configBuilder =
          CrosstoolConfig.CrosstoolRelease.newBuilder();
      TextFormat.merge(crosstoolFileContents, configBuilder);
      List<CToolchain> toolchainList = configBuilder.build().getToolchainList();
      ImmutableList.Builder<CcToolchainConfig> toolchainConfigInfoBuilder = ImmutableList.builder();
      for (CToolchain toolchain : toolchainList) {
        toolchainConfigInfoBuilder.add(
            new CcToolchainConfig(
                toolchain.getTargetCpu(),
                toolchain.getCompiler(),
                toolchain.getToolchainIdentifier(),
                toolchain.getHostSystemName(),
                toolchain.getTargetSystemName(),
                toolchain.getAbiVersion(),
                toolchain.getAbiLibcVersion(),
                toolchain.getTargetLibc(),
                toolchain.getFeatureList().stream()
                    .map(feature -> feature.getName())
                    .collect(ImmutableList.toImmutableList()),
                /* actionConfigs= */ ImmutableList.of(),
                /* artifactNamePatterns= */ ImmutableList.of(),
                /* toolPaths= */ ImmutableList.of()));
      }
      ccToolchainConfigs = toolchainConfigInfoBuilder.build();
    }
    Set<String> seenCpus = new LinkedHashSet<>();
    StringBuilder compilerMap = new StringBuilder();
    for (CcToolchainConfig toolchain : ccToolchainConfigs) {
      String staticRuntimeLabel =
          toolchain.hasStaticLinkCppRuntimesFeature()
              ? "mock-static-runtimes-target-for-" + toolchain.getToolchainIdentifier()
              : null;
      String dynamicRuntimeLabel =
          toolchain.hasStaticLinkCppRuntimesFeature()
              ? "mock-dynamic-runtimes-target-for-" + toolchain.getToolchainIdentifier()
              : null;
      if (staticRuntimeLabel != null) {
        runtimes.add(
            Joiner.on('\n')
                .join(
                    "filegroup(",
                    "  name = '" + staticRuntimeLabel + "',",
                    "  licenses = ['unencumbered'],",
                    "  srcs = ['libstatic-runtime-lib-source.a'])",
                    ""));
      }
      if (dynamicRuntimeLabel != null) {
        runtimes.add(
            Joiner.on('\n')
                .join(
                    "filegroup(",
                    "  name = '" + dynamicRuntimeLabel + "',",
                    "  licenses = ['unencumbered'],",
                    "  srcs = ['libdynamic-runtime-lib-source.so'])",
                    ""));
      }

      // Generate entry to cc_toolchain_suite.toolchains
      if (seenCpus.add(toolchain.getTargetCpu())) {
        compilerMap.append(
            String.format(
                "'%s': ':cc-compiler-%s-%s',\n",
                toolchain.getTargetCpu(), toolchain.getTargetCpu(), toolchain.getCompiler()));
      }
      compilerMap.append(
          String.format(
              "'%s|%s': ':cc-compiler-%s-%s',\n",
              toolchain.getTargetCpu(),
              toolchain.getCompiler(),
              toolchain.getTargetCpu(),
              toolchain.getCompiler()));

      // Generate cc_toolchain target
      String suffix = toolchain.getTargetCpu() + "-" + toolchain.getCompiler();
      compilationTools.append(
          Joiner.on("\n")
              .join(
                  "toolchain(",
                  "  name = 'cc-toolchain-" + suffix + "',",
                  "  toolchain_type = ':toolchain_type',",
                  "  toolchain = ':cc-compiler-" + suffix + "',",
                  ")",
                  disableCrosstool ? toolchain.getCcToolchainConfigRule() : "",
                  "cc_toolchain(",
                  "  name = 'cc-compiler-" + suffix + "',",
                  "  toolchain_identifier = '" + toolchain.getToolchainIdentifier() + "',",
                  disableCrosstool ? "  toolchain_config = ':" + suffix + "_config'," : "",
                  "  output_licenses = ['unencumbered'],",
                  "  module_map = 'crosstool.cppmap',",
                  "  cpu = '" + toolchain.getTargetCpu() + "',",
                  "  compiler = '" + toolchain.getCompiler() + "',",
                  "  ar_files = 'ar-" + toolchain.getTargetCpu() + "',",
                  "  as_files = 'as-" + toolchain.getTargetCpu() + "',",
                  "  compiler_files = 'compile-" + toolchain.getTargetCpu() + "',",
                  "  dwp_files = 'dwp-" + toolchain.getTargetCpu() + "',",
                  "  linker_files = 'link-" + toolchain.getTargetCpu() + "',",
                  "  strip_files = ':every-file',",
                  "  objcopy_files = 'objcopy-" + toolchain.getTargetCpu() + "',",
                  "  all_files = ':every-file',",
                  "  licenses = ['unencumbered'],",
                  supportsHeaderParsing ? "    supports_header_parsing = 1," : "",
                  dynamicRuntimeLabel == null
                      ? ""
                      : "    dynamic_runtime_lib = '" + dynamicRuntimeLabel + "',",
                  staticRuntimeLabel == null
                      ? ""
                      : "    static_runtime_lib = '" + staticRuntimeLabel + "',",
                  ")",
                  ""));
    }

    String build =
        Joiner.on("\n")
            .join(
                "package(default_visibility=['//visibility:public'])",
                "licenses(['restricted'])",
                "",
                disableCrosstool ? "load(':cc_toolchain_config.bzl', 'cc_toolchain_config')" : "",
                "toolchain_type(name = 'toolchain_type')",
                "cc_toolchain_alias(name = 'current_cc_toolchain')",
                "alias(name = 'toolchain', actual = 'everything')",
                "filegroup(name = 'everything-multilib',",
                "          srcs = glob(['" + version + "/**/*'],",
                "              exclude_directories = 1),",
                "          output_licenses = ['unencumbered'])",
                "",
                String.format(
                    "cc_toolchain_suite(name = 'everything', toolchains = {%s})", compilerMap),
                "",
                String.format(
                    "filegroup(name = 'every-file', srcs = ['%s'])",
                    Joiner.on("', '").join(CROSSTOOL_BINARIES)),
                "",
                compilationTools.toString(),
                Joiner.on("\n").join(runtimes),
                "",
                "filegroup(",
                "    name = 'interface_library_builder',",
                "    srcs = ['build_interface_so'],",
                ")",
                // We add an empty :malloc target in case we need it.
                "cc_library(name = 'malloc')");

    config.create(crosstoolTop + "/" + version + "/x86/bin/gcc");
    config.create(crosstoolTop + "/" + version + "/x86/bin/ld");
    config.overwrite(crosstoolTop + "/BUILD", build);
    if (disableCrosstool) {
      config.overwrite(
          crosstoolTop + "/cc_toolchain_config.bzl",
          ResourceLoader.readFromResources(
              "com/google/devtools/build/lib/analysis/mock/cc_toolchain_config.bzl"));
      config.overwrite(
          TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/cpp/cc_toolchain_config_lib.bzl",
          ResourceLoader.readFromResources(
              TestConstants.BAZEL_REPO_PATH + "tools/cpp/cc_toolchain_config_lib.bzl"));
      config.overwrite(
          TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/build_defs/cc/action_names.bzl",
          ResourceLoader.readFromResources(
              TestConstants.BAZEL_REPO_PATH + "tools/build_defs/cc/action_names.bzl"));
      config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/build_defs/cc/BUILD");
    } else {
      config.overwrite(crosstoolTop + "/CROSSTOOL", crosstoolFileContents);
    }
    config.create(crosstoolTop + "/crosstool.cppmap", "module crosstool {}");
  }
}
