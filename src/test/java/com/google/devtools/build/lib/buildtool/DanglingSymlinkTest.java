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
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for dangling symlinks. */
@RunWith(JUnit4.class)
public class DanglingSymlinkTest extends BuildIntegrationTestCase {

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

  /** Tests that bad symlinks for inputs are properly handled. */
  @Test
  public void testCircularSymlinkMidLevel() throws Exception {
    Path fooBuildFile =
        write(
            "foo/BUILD",
            """
            sh_binary(
                name = "foo",
                srcs = ["foo.sh"],
            )

            genrule(
                name = "top",
                srcs = [":foo"],
                outs = ["out"],
                cmd = "touch $@",
            )
            """);
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
            """
            sh_binary(
                name = "foo",
                srcs = ["foo.sh"],
            )

            genrule(
                name = "top",
                srcs = [":foo"],
                outs = ["out"],
                cmd = "touch $@",
            )
            """);
    Path fooShFile = fooBuildFile.getParentDirectory().getRelative("foo.sh");
    fooShFile.createSymbolicLink(PathFragment.create("doesnotexist"));

    assertThrows(BuildFailedException.class, () -> buildTarget("//foo:top"));
    events.assertContainsError("Symlinking //foo:foo failed: missing input file '//foo:foo.sh'");
  }

  @Test
  public void globDanglingSymlink() throws Exception {
    Path packageDirPath = write("foo/BUILD", "exports_files(glob(['*.txt']))").getParentDirectory();
    write("foo/existing.txt");
    Path badSymlink = packageDirPath.getChild("bad.txt");
    FileSystemUtils.ensureSymbolicLink(badSymlink, "nope");
    // Successful build: dangling symlinks in glob are ignored.
    buildTarget("//foo:all");
  }

  @Test
  public void globSymlinkCycle() throws Exception {
    Path fooBuildFile = write("foo/BUILD", "sh_library(name = 'foo', srcs = glob(['*.sh']))");
    fooBuildFile
        .getParentDirectory()
        .getChild("foo.sh")
        .createSymbolicLink(PathFragment.create("foo.sh"));
    TargetParsingException e =
        assertThrows(TargetParsingException.class, () -> buildTarget("//foo:foo"));
    assertThat(e.getDetailedExitCode().getFailureDetail().getPackageLoading().getCode())
        .isEqualTo(FailureDetails.PackageLoading.Code.EVAL_GLOBS_SYMLINK_ERROR);
  }

  @Test
  public void globMissingFile() throws Exception {}
}
