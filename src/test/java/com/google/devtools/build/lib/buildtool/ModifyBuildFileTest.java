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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.buildtool.util.GoogleBuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * This test case tests that we handle things properly when a BUILD file is
 * modified.
 */
@RunWith(JUnit4.class)
public class ModifyBuildFileTest extends GoogleBuildIntegrationTestCase {
  private void writeBuildFileAndSetMtime(long mtime) throws IOException {
    Path buildFile = write("modify_build_file_test/BUILD",
                           "cc_binary(name = 'foo', srcs = ['foo.cc'])");
    buildFile.setLastModifiedTime(mtime);
  }

  private void updateBuildFileAndSetMtime(long mtime)
      throws IOException {
    // put an error in the BUILD file.
    Path buildFile =
        write(
            "modify_build_file_test/BUILD",
            "cc_binary(name = 'foo', doesnotexist = 1, srcs = ['foo.cc'])");
    buildFile.setLastModifiedTime(mtime);
    // other files remain unchanged
  }

  private void writeCcFile() throws IOException {
    write("modify_build_file_test/foo.cc",
        "#include <stdio.h>",
        "int main() {",
        "  printf(\"In modify_build_file_test/foo.cc main()\\n\");",
        "  return 0;",
        "}");
  }

  @Test
  public void testModifyBuildFile() throws Exception {
    //
    // Initial build: this should work
    //
    writeBuildFileAndSetMtime(1000);
    writeCcFile();
    buildTarget("//modify_build_file_test:foo");

    Path executable = getExecutableLocation("//modify_build_file_test:foo");

    String firstOutput = run(executable);
    assertThat(firstOutput).isEqualTo("In modify_build_file_test/foo.cc main()\n");

    //
    // Put a syntax error in the BUILD file and rebuild;
    // this is supposed to fail.
    //
    updateBuildFileAndSetMtime(2000);
    assertThrows(TargetParsingException.class, () -> buildTarget("//modify_build_file_test:foo"));

    //
    // Restore the original contents BUILD file and rebuild;
    // this is supposed to work.
    //
    writeBuildFileAndSetMtime(3000);
    buildTarget("//modify_build_file_test:foo");
    Path executable2 = getExecutableLocation("//modify_build_file_test:foo");
    // the location of the executable shouldn't have changed
    assertThat(executable2).isEqualTo(executable);
    // the output shouldn't have changed

    assertThat(run(executable)).isEqualTo(firstOutput);
  }

  /**
   * Regression test for bug #1953951: .h file left over in genfiles shadows
   * static .h file even if there is no build rule for it anymore.
   */
  @Test
  public void testChangeGeneratedHeaderToNonGenerated() throws Exception {
    // Include test.h created in genrule, should succeed.
    write("x/BUILD",
        "genrule(name = 'gen',",
        "        outs = ['test.h'],",
        "        cmd = 'echo //test > $@')",
        "cc_binary(name = 'bin',",
        "          srcs = ['bin.cc', 'test.h'])");
    write("x/bin.cc",
        "#include \"x/test.h\"",
        "int main() {}");
    buildTarget("//x:bin");

    // Remove genrule, build should fail.
    write("x/BUILD",
        "cc_binary(name = 'bin',",
        "          srcs = ['bin.cc', 'test.h'])");
    assertThrows(BuildFailedException.class, () -> buildTarget("//x:bin"));
    events.assertContainsError("missing input file '//x:test.h'");

    // Make test.h undeclared, build should still fail.
    events.collector().clear();
    write("x/BUILD",
        "cc_binary(name = 'bin',",
        "          srcs = ['bin.cc'])");
    assertThrows(BuildFailedException.class, () -> buildTarget("//x:bin"));
    events.assertContainsError("undeclared inclusion(s) in rule '//x:bin':");

    // Create header source file, build is still undeclared.
    write("x/test.h");
    assertThrows(BuildFailedException.class, () -> buildTarget("//x:bin"));
    events.assertContainsError("undeclared inclusion(s) in rule '//x:bin':");

    // Declare the header, watch the build succeed.
    write("x/BUILD",
        "cc_binary(name = 'bin',",
        "          srcs = ['bin.cc', 'test.h'])");
    buildTarget("//x:bin");
  }

  @Test
  public void testChangeHeaderToGenerated() throws Exception {
    // Should pick up the source header.
    write("x/BUILD",
        "cc_binary(name = 'bin',",
        "          srcs = ['bin.cc', 'test.h'])");
    write("x/test.h",
        "#define VALUE 1");
    write("x/bin.cc",
        "#include <stdio.h>",
        "#include \"x/test.h\"",
        "int main() { printf(\"%d\", VALUE); }");
    assertThat(buildAndRun("//x:bin")).isEqualTo("1");

    // Should pick up the generated header.
    write("x/BUILD",
        "genrule(name = 'gen',",
        "        outs = ['test.h'],",
        "        cmd = 'echo \"#define VALUE 2\" > $@')",
        "cc_binary(name = 'bin',",
        "          srcs = ['bin.cc', 'test.h'])");
    delete("x/test.h");
    assertThat(buildAndRun("//x:bin")).isEqualTo("2");
  }

  private void delete(String path) throws IOException {
    getCommandEnvironment().getWorkspace().getRelative(path).delete();
  }

  private String buildAndRun(String target) throws Exception {
    buildTarget(target);
    return run(getExecutableLocation(target));
  }
}
