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
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
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
  private boolean addEmbeddedRuntimes;
  private String staticRuntimesLabel;
  private String dynamicRuntimesLabel;
  private ImmutableList<String> archs;
  private boolean addModuleMap;
  private boolean supportsHeaderParsing;

  Crosstool(MockToolsConfig config, String crosstoolTop) {
    this.config = config;
    this.crosstoolTop = crosstoolTop;
  }

  public Crosstool setAddModuleMap(boolean addModuleMap) {
    this.addModuleMap = addModuleMap;
    return this;
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

  public Crosstool setEmbeddedRuntimes(
      boolean addEmbeddedRuntimes, String staticRuntimesLabel, String dynamicRuntimesLabel) {
    this.addEmbeddedRuntimes = addEmbeddedRuntimes;
    this.staticRuntimesLabel = staticRuntimesLabel;
    this.dynamicRuntimesLabel = dynamicRuntimesLabel;
    return this;
  }

  public void write() throws IOException {
    String runtimes = "";
    for (String arch : archs) {
      runtimes +=
          Joiner.on('\n')
              .join(
                  "filegroup(name = 'dynamic-runtime-libs-" + arch + "',",
                  "          licenses = ['unencumbered'],",
                  "          srcs = ['libdynamic-runtime-lib-source.so'])",
                  "filegroup(name = 'static-runtime-libs-" + arch + "',",
                  "          licenses = ['unencumbered'],",
                  "          srcs = ['static-runtime-lib-source.a'])\n");
    }

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
    StringBuilder compilerMap = new StringBuilder();
    // Remove duplicates
    Set<Pair<String, String>> keys = new LinkedHashSet<>();
    for (CrosstoolConfig.CToolchain toolchain : configBuilder.build().getToolchainList()) {
      Pair<String, String> key = Pair.of(toolchain.getTargetCpu(), toolchain.getCompiler());
      if (!keys.contains(key)) {
        keys.add(key);
        compilerMap.append(
            String.format(
                "'%s|%s': ':cc-compiler-%s-%s',\n", key.first, key.second, key.first, key.second));
      }
    }

    for (Pair<String, String> key : keys) {
      String cpu = key.first;
      String compiler = key.second;
      String compilerRule;
      String staticRuntimesString =
          staticRuntimesLabel == null ? "" : ", '" + staticRuntimesLabel + "'";
      String dynamicRuntimesString =
          dynamicRuntimesLabel == null ? "" : ", '" + dynamicRuntimesLabel + "'";

      compilerRule =
          Joiner.on("\n")
              .join(
                  "cc_toolchain(",
                  "    name = 'cc-compiler-" + cpu + "-" + compiler + "',",
                  "    output_licenses = ['unencumbered'],",
                  addModuleMap ? "    module_map = 'crosstool.cppmap'," : "",
                  "    cpu = '" + cpu + "',",
                  "    compiler = '" + compiler + "',",
                  "    ar_files = 'ar-" + cpu + "',",
                  "    as_files = 'as-" + cpu + "',",
                  "    compiler_files = 'compile-" + cpu + "',",
                  "    dwp_files = 'dwp-" + cpu + "',",
                  "    linker_files = 'link-" + cpu + "',",
                  "    strip_files = ':every-file',",
                  "    objcopy_files = 'objcopy-" + cpu + "',",
                  "    all_files = ':every-file',",
                  "    licenses = ['unencumbered'],",
                  supportsHeaderParsing ? "    supports_header_parsing = 1," : "",
                  "    dynamic_runtime_libs = ['dynamic-runtime-libs-"
                      + cpu
                      + "'"
                      + dynamicRuntimesString
                      + "],",
                  "    static_runtime_libs = ['static-runtime-libs-"
                      + cpu
                      + "'"
                      + staticRuntimesString
                      + "])");

      compilationTools.append(compilerRule + "\n");
    }

    String build =
        Joiner.on("\n")
            .join(
                "package(default_visibility=['//visibility:public'])",
                "licenses(['restricted'])",
                "",
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
                    "filegroup(name = 'every-file', srcs = ['%s'%s%s])",
                    Joiner.on("', '").join(CROSSTOOL_BINARIES),
                    addEmbeddedRuntimes ? ", ':dynamic-runtime-libs-k8'" : "",
                    addEmbeddedRuntimes ? ", ':static-runtime-libs-k8'" : ""),
                "",
                compilationTools.toString(),
                runtimes,
                "",
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
