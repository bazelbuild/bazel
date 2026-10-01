// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp.proto;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class CcProtoLibraryTest extends BuildViewTestCase {
  @Before
  public void setUp() throws Exception {
    MockProtoSupport.setup(mockToolsConfig);
    scratch.appendFile(
        "third_party/protobuf/BUILD.bazel",
        TestConstants.LOAD_PROTO_LANG_TOOLCHAIN,
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "filegroup(name='license')",
        "genrule(name='protoc_gen', cmd='', executable = True, outs = ['protoc'])",
        "proto_library(",
        "    name = 'any_proto',",
        "    srcs = ['any.proto'],",
        ")",
        "proto_lang_toolchain(",
        "    name = 'cc_toolchain',",
        "    command_line = '--cpp_out=$(OUT)',",
        "    blacklisted_protos = [':any_proto'],",
        "    progress_message = 'Generating C++ proto_library %{label}',",
        "    toolchain_type = '@com_google_protobuf//bazel/private:cc_toolchain_type',",
        ")");
    scratch.appendFile(
        "third_party/protobuf/bazel/private/toolchains/BUILD.bazel",
        """
        toolchain(
            name = "cc_source_toolchain",
            exec_compatible_with = [],
            target_compatible_with = [],
            toolchain = "//:cc_toolchain",
            toolchain_type = "//bazel/private:cc_toolchain_type",
        )
        """);
    scratch.appendFile(
        "third_party/protobuf/MODULE.bazel",
        "register_toolchains('//bazel/private/toolchains:all')");
    invalidatePackages(); // A dash of magic to re-evaluate the WORKSPACE file.
  }

  // TODO(carmi): test blacklisted protos. I don't currently understand what's the wanted behavior.

  @Test
  public void testCcProtoLibraryLoadedThroughMacro() throws Exception {
    if (!analysisMock.isThisBazel()) {
      return;
    }
    setupTestCcProtoLibraryLoadedThroughMacro(/* loadMacro= */ true);
    assertThat(getConfiguredTarget("//a:a")).isNotNull();
    assertNoEvents();
  }

  private void setupTestCcProtoLibraryLoadedThroughMacro(boolean loadMacro) throws Exception {
    scratch.file(
        "a/BUILD",
        getAnalysisMock().ccSupport().getMacroLoadStatement(loadMacro, "cc_proto_library"),
        "load('@com_google_protobuf//bazel:proto_library.bzl', 'proto_library')",
        "load('@com_google_protobuf//bazel:cc_proto_library.bzl', 'cc_proto_library')",
        "cc_proto_library(",
        "    name='a',",
        "    deps=[':a_p'],",
        ")",
        "proto_library(",
        "    name='a_p',",
        "    srcs = ['a.proto'],",
        ")");
  }
}
