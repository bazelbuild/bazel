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

package com.google.devtools.build.lib.packages.util;

import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;

/** Mock python support in Bazel. */
public final class BazelMockPythonSupport extends MockPythonSupport {

  public static final BazelMockPythonSupport INSTANCE = new BazelMockPythonSupport();

  private BazelMockPythonSupport() {}

  private static void addTool(MockToolsConfig config, String toolRelativePath) throws IOException {
    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + toolRelativePath,
        ResourceLoader.readFromResources(TestConstants.BAZEL_REPO_PATH + toolRelativePath));
  }

  @Override
  public void setup(MockToolsConfig config) throws IOException {
    writeMacroFile(config);

    addTool(config, "tools/python/python_version.bzl");
    addTool(config, "tools/python/srcs_version.bzl");
    addTool(config, "tools/python/toolchain.bzl");
    addTool(config, "tools/python/utils.bzl");
    addTool(config, "tools/python/private/defs.bzl");

    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/python/BUILD",
        "package(default_visibility=['//visibility:public'])",
        getMacroLoadStatement(),
        "load(':python_version.bzl', 'define_python_version_flag')",
        "load('//tools/python:toolchain.bzl', 'py_runtime_pair')",
        "define_python_version_flag(",
        "    name = 'python_version',",
        ")",
        "config_setting(",
        "    name = 'PY2',",
        "    flag_values = {':python_version': 'PY2'},",
        ")",
        "config_setting(",
        "    name = 'PY3',",
        "    flag_values = {':python_version': 'PY3'},",
        ")",
        "toolchain_type(name = 'toolchain_type')",
        "constraint_setting(name = 'py2_interpreter_path')",
        "constraint_setting(name = 'py3_interpreter_path')",
        "py_runtime(",
        "    name = 'py2_interpreter',",
        "    interpreter_path = '/usr/bin/mockpython2',",
        "    python_version = 'PY2',",
        ")",
        "py_runtime(",
        "    name = 'py3_interpreter',",
        "    interpreter_path = '/usr/bin/mockpython3',",
        "    python_version = 'PY3',",
        ")",
        "py_runtime_pair(",
        "    name = 'default_py_runtime_pair',",
        "    py2_runtime = ':py2_interpreter',",
        "    py3_runtime = ':py3_interpreter',",
        ")",
        "toolchain(",
        "    # The Python workspace suffix looks to register a toolchain of this name.",
        "    name = 'autodetecting_toolchain',",
        "    toolchain = ':default_py_runtime_pair',",
        "    toolchain_type = ':toolchain_type',",
        ")",
        "exports_files(['precompile.py'])",
        "sh_binary(name='2to3', srcs=['2to3.sh'])");
  }

  @Override
  public String createPythonTopEntryPoint(MockToolsConfig config, String pyRuntimeLabel)
      throws IOException {
    // Under BazelPythonSemantics, we can simply set --python_top to be the py_runtime target.
    return pyRuntimeLabel;
  }
}
