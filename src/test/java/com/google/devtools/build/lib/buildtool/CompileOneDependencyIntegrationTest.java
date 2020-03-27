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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertContainsEvent;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.buildtool.util.GoogleBuildIntegrationTestCase;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test the semantics of the compile_one_dependency flag: for each command-line argument (which must
 * be a source file path relative to the workspace) rebuild a single target that depends on it.
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class CompileOneDependencyIntegrationTest extends GoogleBuildIntegrationTestCase {

  @Test
  public void testCcBinaryMember() throws Exception {
    EventCollector eventCollector = new EventCollector(EventKind.START);
    events.addHandler(eventCollector);
    addOptions("--compile_one_dependency");

    write("package/BUILD",
          "cc_binary(name='foo', srcs=['foo.cc'], malloc = '//base:system_malloc')");
    write("package/foo.cc",
          "#include <stdio.h>",
          "int main() {",
          "  printf(\"Hello, world!\\n\");",
          "  return 0;",
          "}");
    buildTarget("package/foo.cc");

    // Check that 'foo' is recompiled and linked.
    assertContainsEvent(eventCollector, "Compiling package/foo.cc");
    assertContainsEvent(eventCollector, "Linking package/foo");
  }

  /**
   * Regression test for b/13394215: --compile_one_dependency with --keep_going not working in
   * skyframe.
   */
  @Test
  public void testBadPackage() throws Exception {
    addOptions("--compile_one_dependency", "--keep_going");

    write("package/BUILD",
          "cc_binary(name='foo', srcs=['foo.cc'], malloc = '//base:system_malloc')",
          "invalidbuildsyntax");
    write("package/foo.cc",
          "#include <stdio.h>",
          "int main() {",
          "  printf(\"Hello, world!\\n\");",
          "  return 0;",
          "}");
    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildTarget("package/foo.cc"));
    assertThat(e)
        .hasMessageThat()
        .contains("command succeeded, but there were errors parsing the target pattern");
  }

  @Test
  public void testTransitiveRdepFromSourceFileViaChainOfFilegroupsThatGoesThroughPackageInError()
      throws Exception {
    addOptions("--compile_one_dependency", "--keep_going", "--noanalyze");

    write(
        "package/BUILD",
        "exports_files(['foo.cc'])",
        "cc_binary(name='foo', srcs=['fg'], malloc = '//base:system_malloc')",
        "filegroup(name = 'fg', srcs = ['//brokenpackage:fg'])");
    write(
        "package/foo.cc",
        "#include <stdio.h>",
        "int main() {",
        "  printf(\"Hello, world!\\n\");",
        "  return 0;",
        "}");
    write("brokenpackage/BUILD", "filegroup(name = 'fg', srcs = ['//package:foo.cc'])", "nope");
    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildTarget("package:foo.cc"));
    assertThat(e)
        .hasMessageThat()
        .contains("command succeeded, but there were errors parsing the target pattern");
  }
}
