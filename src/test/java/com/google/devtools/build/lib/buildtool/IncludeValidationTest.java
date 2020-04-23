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

import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.buildtool.util.GoogleBuildIntegrationTestCase;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * End-to-end or integration test of some of the tricky include validation
 * cases, including regressions.
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class IncludeValidationTest extends GoogleBuildIntegrationTestCase {

  private void writeFooBuild(boolean withDeps) throws Exception {
    write("foo/BUILD",
        "genrule(name = 'gen',",
        "        srcs = ['foo.h'],",
        "        outs = ['foo_gen.h'],",
        "        cmd = '/bin/cp $(location foo.h) $(location foo_gen.h)')",
        "cc_binary(name = 'foo', srcs = [ 'foo.cc', ",
        withDeps ? " ':foo_gen.h']," : "],",
        "  malloc = '//base:system_malloc')");
  }

  private void writeFooSource() throws Exception {
    write("foo/foo.h",
        "//empty");
    write("foo/foo.cc",
        "#include \"foo/foo_gen.h\"",
        "int main(int argc, char** argv) { return 0; }");
  }

  @Test
  public void testDroppedDependency() throws Exception {
    writeFooBuild(true);
    writeFooSource();
    buildTarget("//foo:foo");

    writeFooBuild(false);
    assertThrows(BuildFailedException.class, () -> buildTarget("//foo:foo"));
  }
}
