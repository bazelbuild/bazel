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
 * A helper class to create a crosstool package containing a CROSSTOOL file, and the various
 * rules needed for a mock - use this only for configured target tests, not for execution tests.
 */
final class Crosstool {
  private static final ImmutableList<String> CROSSTOOL_BINARIES =
      ImmutableList.of("ar", "as", "compile", "dwp", "link", "objcopy", "llvm-profdata");

  private final MockToolsConfig config;

  private final String crosstoolTop;
  private String version;
  private String crosstoolFileContents;
  private ImmutableList<String> archs;
  private boolean supportsHeaderParsing;

  Crosstool(MockToolsConfig config, String crosstoolTop) {
    this.config = config;
    this.crosstoolTop = crosstoolTop;
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

    CrosstoolConfig.CrosstoolRelease.Builder configBuilder =
        CrosstoolConfig.CrosstoolRelease.newBuilder();
    TextFormat.merge(crosstoolFileContents, configBuilder);

    List<CToolchain> toolchainList = configBuilder.build().getToolchainList();
    Set<String> seenCpus = new LinkedHashSet<>();
    StringBuilder compilerMap = new StringBuilder();
    for (CToolchain toolchain : toolchainList) {
      boolean hasStaticLinkCppRuntimesFeature =
          toolchain.getFeatureList().stream()
              .anyMatch(f -> f.getName().equals(CppRuleClasses.STATIC_LINK_CPP_RUNTIMES));
      String staticRuntimeLabel =
          hasStaticLinkCppRuntimesFeature
              ? "mock-static-runtimes-target-for-" + toolchain.getToolchainIdentifier()
              : null;
      String dynamicRuntimeLabel =
          hasStaticLinkCppRuntimesFeature
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
                  "cc_toolchain(",
                  "  name = 'cc-compiler-" + suffix + "',",
                  "  toolchain_identifier = '" + toolchain.getToolchainIdentifier() + "',",
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
    config.getPath(crosstoolTop + "/CROSSTOOL");
    config.overwrite(crosstoolTop + "/BUILD", build);
    config.overwrite(crosstoolTop + "/CROSSTOOL", crosstoolFileContents);
    config.create(crosstoolTop + "/crosstool.cppmap", "module crosstool {}");
  }
}
