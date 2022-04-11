// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.view.go;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.CompileOnlyTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests that validate --compile_only behavior. */
@RunWith(JUnit4.class)
public class GoCompileOnlyTest extends CompileOnlyTestCase {

  @Test
  public void testGoCompileOnly() throws Exception {
    scratch.file(
        "go_test/main/BUILD",
        "go_binary(",
        "    name = 'main',",
        "    deps = ['//go_test/hello'],",
        "    srcs = ['main.go'],",
        ")");
    scratch.file(
        "go_test/hello/BUILD",
        "go_test(",
        "    name = 'hello_test',",
        "    srcs = [",
        "        'hello_test.go'",
        "    ],",
        "    library = ':hello'",
        ")",
        "go_library(",
        "    name = 'hello',",
        "    srcs = [",
        "        'hello.go'",
        "    ],",
        ")");
    scratch.file(
        "go_test/main/main.go",
        "package main",
        "import \"go_test/hello/hello\"",
        "func main() {",
        "  hello.Hello();",
        "}");
    scratch.file(
        "go/hello/hello.go",
        "package hello;",
        "",
        "func Hello() {",
        "  fmt.Println(`Hello!`)",
        "}");
    scratch.file(
        "go_test/hello/hello_test.go",
        "package hello_test",
        "import \"testing\"",
        "func TestHello(t *testing.T) {}");

    ConfiguredTarget binaryTarget = getConfiguredTarget("//go_test/main:main");
    // Check that only the package been compiled and not the linked binary.
    assertThat(getArtifactByExecPathSuffix(binaryTarget, "/main.a")).isNotNull();
    assertThat(getArtifactByExecPathSuffix(binaryTarget, "/main")).isNull();

    ConfiguredTarget libTarget = getConfiguredTarget("//go_test/hello:hello");
    assertThat(getArtifactByExecPathSuffix(libTarget, "/hello.a")).isNotNull();

    ConfiguredTarget testTarget = getConfiguredTarget("//go_test/hello:hello_test");
    assertThat(getArtifactByExecPathSuffix(testTarget, "/hello_test_testlib.a")).isNotNull();
  }
}
