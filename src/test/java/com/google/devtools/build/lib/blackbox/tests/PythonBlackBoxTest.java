// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.tests;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.blackbox.bazel.PythonToolsSetup;
import com.google.devtools.build.lib.blackbox.framework.BlackBoxTestEnvironment;
import com.google.devtools.build.lib.blackbox.framework.BuilderRunner;
import com.google.devtools.build.lib.blackbox.framework.ProcessResult;
import com.google.devtools.build.lib.blackbox.framework.ToolsSetup;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Test;

/** End to end tests for building and running Python targets. */
public class PythonBlackBoxTest extends AbstractBlackBoxTest {
  private static final String HELLO = "Hello, World!";

  @Override
  protected ImmutableList<ToolsSetup> getAdditionalTools() {
    return ImmutableList.of((new PythonToolsSetup()));
  }

  @Test
  public void testCompileAndRunHelloWorldStub() throws Exception {
    writeHelloWorldFiles();

    BuilderRunner bazel = context().bazel();
    bazel.build("//python/hello:hello");

    ProcessResult result = context().runBuiltBinary(bazel, "python/hello/hello", -1);
    assertThat(result.outString()).isEqualTo(HELLO);

    Path binaryPath = context().resolveBinPath(bazel, "python/hello/hello.par");
    assertThat(Files.exists(binaryPath)).isFalse();
  }

  private void writeHelloWorldFiles() throws IOException {
    context().write("python/hello/BUILD", "py_binary(name = 'hello', srcs = ['hello.py'])");
    context().write("python/hello/hello.py", String.format("print ('%s')", HELLO));
  }
}
