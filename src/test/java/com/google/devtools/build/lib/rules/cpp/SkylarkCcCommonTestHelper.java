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

import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.testutil.Scratch;

/** Methods useful for tests testing the C++ Skylark API. */
public final class SkylarkCcCommonTestHelper {

  public static final String CC_SKYLARK_WHITELIST_FLAG =
      "--experimental_cc_skylark_api_enabled_packages=tools/build_defs,experimental";

  public static void createFilesForTestingCompilation(
      Scratch scratch, String bzlFilePath, String compileProviderLines) throws Exception {
    createFiles(scratch, bzlFilePath, compileProviderLines, "");
  }

  public static void createFilesForTestingLinking(
      Scratch scratch, String bzlFilePath, String linkProviderLines) throws Exception {
    createFiles(scratch, bzlFilePath, "", linkProviderLines);
  }

  public static void createFiles(Scratch scratch, String bzlFilePath) throws Exception {
    createFiles(scratch, bzlFilePath, "", "");
  }

  public static void createFiles(
      Scratch scratch, String bzlFilePath, String compileProviderLines, String linkProviderLines)
      throws Exception {
    String fragments = "    fragments = ['google_cpp', 'cpp'],";
    if (AnalysisMock.get().isThisBazel()) {
      fragments = "    fragments = ['cpp'],";
    }
    scratch.overwriteFile(bzlFilePath + "/BUILD");
    scratch.file(
        bzlFilePath + "/extension.bzl",
        "def _cc_skylark_library_impl(ctx):",
        "    dep_linking_contexts = []",
        "    for dep in ctx.attr._deps:",
        "        dep_linking_contexts.append(dep[CcInfo].linking_context)",
        "    toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "    feature_configuration = cc_common.configure_features(",
        "        cc_toolchain=toolchain,",
        "        requested_features = ctx.features,",
        "        unsupported_features = ctx.disabled_features)",
        "    compilation_info = cc_common.compile(",
        "        ctx=ctx,",
        "        feature_configuration=feature_configuration,",
        "        cc_toolchain=toolchain,",
        "        srcs=ctx.files.srcs,",
        "        hdrs=ctx.files.hdrs" + (compileProviderLines.isEmpty() ? "" : ","),
        "        " + compileProviderLines,
        "    )",
        "    linking_info = cc_common.link(",
        "        ctx=ctx,",
        "        feature_configuration=feature_configuration,",
        "        cc_compilation_outputs=compilation_info.cc_compilation_outputs,",
        "        cc_toolchain=toolchain" + (linkProviderLines.isEmpty() ? "" : ","),
        "        " + linkProviderLines,
        "    )",
        "    files_to_build = []",
        "    files_to_build.extend(compilation_info.cc_compilation_outputs",
        "      .object_files(use_pic=True))",
        "    files_to_build.extend(compilation_info.cc_compilation_outputs",
        "      .object_files(use_pic=False))",
        "    library_to_link = linking_info.cc_linking_outputs.library_to_link",
        "    if library_to_link.pic_static_library != None:",
        "       files_to_build.append(library_to_link.pic_static_library)",
        "    files_to_build.append(library_to_link.dynamic_library)",
        "    return struct(",
        "            libraries=[library_to_link],",
        "            providers=[DefaultInfo(files=depset(files_to_build)),",
        "            CcInfo(compilation_context=compilation_info.compilation_context,",
        "            linking_context=linking_info.linking_context)])",
        "cc_skylark_library = rule(",
        "    implementation = _cc_skylark_library_impl,",
        "    attrs = {",
        "      'srcs': attr.label_list(allow_files=True),",
        "      'hdrs': attr.label_list(allow_files=True),",
        "      '_deps': attr.label_list(default=['//foo:dep1', '//foo:dep2']),",
        "      '_cc_toolchain': attr.label(default =",
        "          configuration_field(fragment = 'cpp', name = 'cc_toolchain'))",
        "    },",
        fragments,
        ")");
    scratch.file(
        "foo/BUILD",
        "load('//" + bzlFilePath + ":extension.bzl', 'cc_skylark_library')",
        "cc_library(",
        "    name = 'dep1',",
        "    srcs = ['dep1.cc'],",
        "    hdrs = ['dep1.h'],",
        "    linkopts = ['-DEP1_LINKOPT'],",
        ")",
        "cc_library(",
        "    name = 'dep2',",
        "    srcs = ['dep2.cc'],",
        "    hdrs = ['dep2.h'],",
        "    linkopts = ['-DEP2_LINKOPT'],",
        ")",
        "cc_skylark_library(",
        "    name = 'skylark_lib',",
        "    srcs = ['skylark_lib.cc'],",
        "    hdrs = ['skylark_lib.h'],",
        ")",
        "cc_binary(",
        "    name = 'bin',",
        "    deps = ['skylark_lib'],",
        ")");
  }
}
