// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.view.python;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests PythonRuntimeAlias class */
@RunWith(JUnit4.class)
public class PythonRuntimeAliasTest extends BuildViewTestCase {

  @Test
  public void testPythonRuntimeAlias() throws Exception {
    scratch.file("a/BUILD", "python_runtime_alias(name='current_python_runtime')");

    ConfiguredTarget target = getConfiguredTarget("//a:current_python_runtime");

    assertThat(target.getLabel().toString()).isEqualTo("//tools/python:default_python_runtime");
  }

  @Test
  public void testPythonRuntimeAliasWithPythonTopArgument() throws Exception {
    scratch.file("a/BUILD", "python_runtime_alias(name='current_python_runtime')");
    scratch.file("b/BUILD", "filegroup(name = 'another_python_top', srcs = [])");
    useConfiguration("--python_top=//b:another_python_top");

    ConfiguredTarget target = getConfiguredTarget("//a:current_python_runtime");

    assertThat(target.getLabel().toString()).isEqualTo("//b:another_python_top");
  }

  @Test
  public void testFileProviderInPythonRuntimeAlias() throws Exception {
    scratch.file("a/BUILD", "python_runtime_alias(name='current_python_runtime')");
    scratch.file(
        "b/BUILD",
        "filegroup(",
        "    name = 'another_python_top',",
        "    srcs = ['some.file1', 'some.file2']",
        ")");
    useConfiguration("--python_top=//b:another_python_top");

    ConfiguredTarget target = getConfiguredTarget("//a:current_python_runtime");
    FileProvider fileProvider = target.getProvider(FileProvider.class);

    assertThat(fileProvider).isNotNull();
    assertThat(prettyArtifactNames(fileProvider.getFilesToBuild()))
        .containsExactly("b/some.file1", "b/some.file2");
  }
}
