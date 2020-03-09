// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;

/**
 * Creates mock BUILD files required for J2Objc.
 */
public final class MockJ2ObjcSupport {
  /**
   * Setup the support for building with J2ObjC.
   */
  public static void setup(MockToolsConfig config) throws IOException {
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/java/j2objc/jre_emul.jar");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/java/j2objc/mod/release");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/java/j2objc/mod/lib/mods");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/java/j2objc/jre.h");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/java/j2objc/jre.m");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/java/j2objc/runtime.h");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/java/j2objc/runtime.m");
    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/java/j2objc/proto_plugin_binary");
    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/java/j2objc/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "licenses(['notice'])",
        "",
        "exports_files(['jre_emul.jar'])",
        "",
        "filegroup(",
        "    name = 'jre_emul_module',",
        "    srcs = ['mod/release', 'mod/lib/mods'])",
        "",
        "objc_library(",
        "    name = 'jre_emul_lib',",
        "    hdrs = ['jre_emul.h'],",
        "    srcs = ['jre_emul.m'],",
        "    deps = [':jre_core_lib', ':jre_io_lib'],",
        "    tags = ['j2objc_jre_lib'])",
        "",
        "objc_library(",
        "    name = 'jre_core_lib',",
        "    hdrs = ['jre_core.h'],",
        "    srcs = ['jre_core.m'],",
        "    tags = ['j2objc_jre_lib'])",
        "",
        "objc_library(",
        "    name = 'jre_io_lib',",
        "    hdrs = ['jre_io.h'],",
        "    srcs = ['jre_io.m'],",
        "    deps = [':jre_core_lib'],",
        "    tags = ['j2objc_jre_lib'])",
        "",
        "objc_library(",
        "    name = 'proto_runtime',",
        "    hdrs = ['runtime.h'],",
        "    srcs = ['runtime.m'])",
        "",
        "filegroup(",
        "    name = 'proto_plugin',",
        "    srcs = ['proto_plugin_binary'])");

    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/j2objc/BUILD",
        TestConstants.LOAD_PROTO_LANG_TOOLCHAIN,
        "package(default_visibility=['//visibility:public'])",
        "licenses(['notice'])",
        "filegroup(",
        "    name = 'j2objc_wrapper',",
        "    srcs = ['j2objc_wrapper.py'])",
        "filegroup(",
        "    name = 'j2objc_header_map',",
        "    srcs = ['j2objc_header_map.py'])",
        "proto_lang_toolchain(",
        "    name = 'j2objc_proto_toolchain',",
        "    blacklisted_protos = [':j2objc_proto_blacklist'],",
        "    command_line = '--PLUGIN_j2objc_out=file_dir_mapping,generate_class_mappings:$(OUT)',",
        "    visibility = ['//visibility:public'],",
        "    plugin = '//third_party/java/j2objc:proto_plugin',",
        "    runtime = '//third_party/java/j2objc:proto_runtime',",
        ")",
        "exports_files(['j2objc_deploy.jar'])",
        "proto_library(",
        "    name = 'j2objc_proto_blacklist',",
        "    deps = [",
        "        '" + TestConstants.TOOLS_REPOSITORY + "//tools/j2objc/proto:blacklisted'",
        "    ])");

    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/j2objc/proto/BUILD",
        TestConstants.LOAD_PROTO_LANG_TOOLCHAIN,
        "package(default_visibility=['//visibility:public'])",
        "proto_library(name = 'blacklisted',",
        "              srcs = ['blacklisted.proto'])");

    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/j2objc/proto/blacklisted.proto");

    if (config.isRealFileSystem()) {
      config.linkTool(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/j2objc/j2objc_deploy.jar");
      config.linkTool(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/j2objc/j2objc_wrapper.py");
      config.linkTool(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/j2objc/j2objc_header_map.py");
    } else {
      config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/j2objc/j2objc_deploy.jar");
      config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/j2objc/j2objc_wrapper.py");
      config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/j2objc/j2objc_header_map.py");
    }
  }
}
