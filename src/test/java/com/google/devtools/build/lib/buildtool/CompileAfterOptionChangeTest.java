// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test of compilation of involving change command line arguments. */
@RunWith(JUnit4.class)
public class CompileAfterOptionChangeTest extends BuildIntegrationTestCase {

  private void writeSourceFiles() throws IOException {
    // A program that uses a command line argument to the compilation to
    // create its output
    write("pkg/hello.cc",
          "#include <stdio.h>",
          "#ifndef GREETING",
          "#define GREETING DEFAULT_GREETING",
          "#endif",
          "extern void printf(const char *s);",
          "int main() {",
          "  printf(\"%s\", GREETING \", World!\");",
          "}");
  }

  /**
   * Adds the specified options and builds the application.
   * Note that any options set in buildApp() are sticky for subsequent calls
   * to buildApp() from the same test method.
   */
  private BuildResult buildApp(String... requestArgs) throws Exception {
    addOptions(requestArgs);
    return buildTarget("//pkg:hello");
  }

  /**
   * Test of minor variations between actions. We pass command line options to
   * gcc and test if we correctly rebuild the application
   */
  @Test
  public void testChangingCommandLineOptionRebuilds() throws Exception {
    writeSourceFiles();
    write(
        "pkg/BUILD",
        """
        cc_binary(
            name = "hello",
            srcs = ["hello.cc"],
            defines = ['DEFAULT_GREETING=\\\\\\"Hello\\\\\\"'],
            malloc = "//base:system_malloc",
        )
        """);
    // Here's why we need so many backslashes:
    //   Java source:            "...'DEFAULT_GREETING=\\\\\\\"Hello\\\\\\\"'..."
    //   BUILD file:              ...'DEFAULT_GREETING=\\\"Hello\\\"'...
    //   Makefile:                ... -DDEFAULT_GREETING=\"Hello\" ...
    //   Arguments passed to sh:  ... -DDEFAULT_GREETING=\"Hello\" ...
    //   Arguments passed to gcc: ... -DDEFAULT_GREETING="Hello" ...
    // Blaze doesn't go through the "Makefile" or "sh" stages, but Blaze's
    // treatment of backslash escapes is compatible with make-dbg, which does.

    ////////////////////////////////////////////////////////////////////////
    // (1) Build using default request options:
    buildApp();
    Path hello = getExecutableLocation("//pkg:hello");

    // Check the output of 'hello':
    assertThat(run(hello)).isEqualTo("Hello, World!");

    ////////////////////////////////////////////////////////////////////////
    // (2) Build again using a different cc_binary rule
    write(
        "pkg/BUILD",
        """
        cc_binary(
            name = "hello",
            srcs = ["hello.cc"],
            defines = ['DEFAULT_GREETING=\\'"Hello again"\\''],
            malloc = "//base:system_malloc",
        )
        """);
    // Here's why we need so many quotes and backslashes:
    //   Java source:            "...'DEFAULT_GREETING=\\'\"Hello again\"\\''..."
    //   BUILD file:              ...'DEFAULT_GREETING=\'"Hello again"\''...
    //  (Makefile:                ... -DDEFAULT_GREETING='"Hello again"' ...)
    //  (Arguments passed to sh:  ... -DDEFAULT_GREETING='"Hello again"' ...)
    //   Arguments passed to gcc: ... -DDEFAULT_GREETING="Hello again" ...
    // Blaze doesn't go through the "Makefile" or "sh" stages, but Blaze's
    // treatment of backslash escapes is compatible with make-dbg, which does.

    buildApp();

    // Check that the output of 'hello' was affected by changing the rule
    assertThat(run(hello)).isEqualTo("Hello again, World!");

    ////////////////////////////////////////////////////////////////////////
    // (3) Build again using additional command line options:
    write(
        "pkg/BUILD",
        """
        cc_binary(
            name = "hello",
            srcs = ["hello.cc"],
            malloc = "//base:system_malloc",
        )
        """);

    buildApp("--copt", "-DGREETING=\"Hi\"");

    // Check that the output of 'hello' was affected by adding the request
    // options:
    assertThat(run(hello)).isEqualTo("Hi, World!");

    ////////////////////////////////////////////////////////////////////////
    // (4) Build again using different command line options:

    // We need the -U option to override the -D option set by the
    // previous call to buildApp() earlier in this method.
    buildApp("--copt", "-UGREETING", "--copt", "-DGREETING=\"Goodbye\"");
    // Check that the output of 'hello' was affected by changing the request
    // options:
    assertThat(run(hello)).isEqualTo("Goodbye, World!");
  }

}
