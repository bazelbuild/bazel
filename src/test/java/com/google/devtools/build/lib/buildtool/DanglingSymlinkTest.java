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
import com.google.devtools.build.lib.packages.util.MockGenruleSupport;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for dangling symlinks. */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class DanglingSymlinkTest extends GoogleBuildIntegrationTestCase {

  @Before
  public final void addNoJobsOption() throws Exception  {
    addOptions("--jobs", "1");
  }

  /**
   * Regression test for bug 823903 about symlink to non-existent target
   * breaking DependencyChecker.
   */
  @Test
  public void testDanglingSymlinks() throws Exception {
    MockGenruleSupport.setup(mockToolsConfig);
    write("test/BUILD",
          "genrule(name='test_ln', srcs=[], outs=['test.out']," +
          " cmd='/bin/ln -sf wrong.out $(@D)/test.out')\n");

    addOptions("--keep_going");
    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildTarget("//test:test_ln"));
    assertThat(e).hasMessageThat().isNull();

    events.assertContainsError("output 'test/test.out' is a dangling symbolic link");
    events.assertContainsError(
        "Executing genrule //test:test_ln failed: not all outputs were created");
  }

  /**
   * Regression test for bug 2411632: cc_library with *.so in srcs list doesn't
   * work as expected.
   */
  @Test
  public void testGeneratedLibs() throws Exception {
    MockGenruleSupport.setup(mockToolsConfig);
    write("test/BUILD",
        "cc_library(name = 'a',",
        "           srcs = ['a.cc'])",
        "genrule(name = 'b',",
        "        srcs = ['liba.so'],",
        "        outs = ['libb.so'],",
        "        cmd = 'cp $(SRCS) $@')",
        "cc_library(name = 'c',",
        "           srcs = [':b'])",
        "cc_binary(name = 'd',",
        "          srcs = ['d.cc'],",
        "          deps = [':c'])");
    write("test/a.cc");
    write("test/d.cc", "int main() { return 0; }");

    addOptions("--jobs=2");

    buildTarget("//test:d");
  }

  /** Tests that bad symlinks for inputs are properly handled. */
  @Test
  public void testCircularSymlinkMidLevel() throws Exception {
    Path fooBuildFile =
        write(
            "foo/BUILD",
            "sh_binary(name = 'foo', srcs = ['foo.sh'])",
            "genrule(name = 'top', srcs = [':foo'], outs = ['out'], cmd = 'touch $@')");
    Path fooShFile = fooBuildFile.getParentDirectory().getRelative("foo.sh");
    fooShFile.createSymbolicLink(PathFragment.create("foo.sh"));

    assertThrows(BuildFailedException.class, () -> buildTarget("//foo:top"));
    events.assertContainsError(
        "Executing genrule //foo:top failed: error reading file '//foo:foo.sh': Symlink cycle");
  }

  @Test
  public void testDanglingSymlinkMidLevel() throws Exception {
    Path fooBuildFile =
        write(
            "foo/BUILD",
            "sh_binary(name = 'foo', srcs = ['foo.sh'])",
            "genrule(name = 'top', srcs = [':foo'], outs = ['out'], cmd = 'touch $@')");
    Path fooShFile = fooBuildFile.getParentDirectory().getRelative("foo.sh");
    fooShFile.createSymbolicLink(PathFragment.create("doesnotexist"));

    assertThrows(BuildFailedException.class, () -> buildTarget("//foo:top"));
    events.assertContainsError("Symlinking //foo:foo failed: missing input file '//foo:foo.sh'");
  }
}
