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
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * A helper class to create a crosstool package containing a CROSSTOOL file, and the various
 * rules needed for a mock - use this only for configured target tests, not for execution tests.
 */
final class Crosstool {
  private static final ImmutableList<String> CROSSTOOL_BINARIES =
      ImmutableList.of("compile", "dwp", "link", "objcopy");

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

    List<String> compilerRules = Lists.newArrayList();

    for (String arch : archs) {
      String compilerRule;
      String staticRuntimesString =
          staticRuntimesLabel == null ? "" : ", '" + staticRuntimesLabel + "'";
      String dynamicRuntimesString =
          dynamicRuntimesLabel == null ? "" : ", '" + dynamicRuntimesLabel + "'";

      compilerRule =
          Joiner.on("\n")
              .join(
                  "cc_toolchain(",
                  "    name = 'cc-compiler-" + arch + "',",
                  "    output_licenses = ['unencumbered'],",
                  addModuleMap ? "    module_map = 'crosstool.cppmap'," : "",
                  "    cpu = '" + arch + "',",
                  "    compiler_files = 'compile-" + arch + "',",
                  "    dwp_files = 'dwp-" + arch + "',",
                  "    linker_files = 'link-" + arch + "',",
                  "    strip_files = ':every-file',",
                  "    objcopy_files = 'objcopy-" + arch + "',",
                  "    all_files = ':every-file',",
                  "    licenses = ['unencumbered'],",
                  supportsHeaderParsing ? "    supports_header_parsing = 1," : "",
                  "    dynamic_runtime_libs = ['dynamic-runtime-libs-"
                      + arch
                      + "'"
                      + dynamicRuntimesString
                      + "],",
                  "    static_runtime_libs = ['static-runtime-libs-"
                      + arch
                      + "'"
                      + staticRuntimesString
                      + "])");

      compilationTools.append(compilerRule + "\n");
      compilerRules.add(":cc-compiler-" + arch);
    }

    String build =
        Joiner.on("\n")
            .join(
                "package(default_visibility=['//visibility:public'])",
                "licenses(['restricted'])",
                "",
                "filegroup(name = 'everything-multilib',",
                "          srcs = glob(['" + version + "/**/*'],",
                "              exclude_directories = 1),",
                "          output_licenses = ['unencumbered'])",
                "",
                String.format(
                    "filegroup(name = 'everything', srcs = ['%s', ':every-file'])",
                    Joiner.on("', '").join(compilerRules)),
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
    config.create(crosstoolTop + "/BUILD", build);
    Path crosstoolPath = config.getPath(crosstoolTop + "/CROSSTOOL");
    if (crosstoolPath.exists()) {
      crosstoolPath.delete();
    }
    config.create(crosstoolTop + "/CROSSTOOL", crosstoolFileContents);
    config.create(crosstoolTop + "/crosstool.cppmap", "module crosstool {}");
  }
}
