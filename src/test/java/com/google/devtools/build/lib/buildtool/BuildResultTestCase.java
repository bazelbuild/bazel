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
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests correctness of the build summary output produced by BuildTool.showBuildResult() method. */
public abstract class BuildResultTestCase extends BuildIntegrationTestCase {
  private RecordingOutErr recOutErr = new RecordingOutErr();

  private static final String GENRULE_ERROR = "Exit 42";

  /**
   * Hook for subclasses to define which executor we use.  (The two concrete
   * subclasses, {Sequential,Parallel}BuildResultTest, are at the bottom of this
   * source file.)
   */
  protected abstract int numJobs();

  @Before
  public final void addOptions() throws Exception  {
    this.outErr = recOutErr;
    addOptions("--jobs", "" + numJobs(), "--show_result", "1000");
  }

  private void build(boolean expectFailure, String expectedError, String... targets)
      throws Exception {
    try {
      buildTarget(targets);
      if (expectFailure) {
        fail();
      }
    } catch (BuildFailedException e) {
      if (e.getMessage() != null) {
        assertThat(e).hasMessageThat().containsMatch(expectedError);
      }
    } finally {
      OutErr.SYSTEM_OUT_ERR.printErr(recOutErr.errAsLatin1());
      OutErr.SYSTEM_OUT_ERR.printOut(recOutErr.outAsLatin1());
    }
  }

  @Test
  public void testKeepGoingResult() throws Exception {
    write("test/BUILD",
        "genrule(name='A', srcs=['A1','A2'], outs=['A.out']," +
        " cmd='/bin/cp test/in $(location A.out)')\n" +
        "genrule(name='A1', srcs=[], outs=['A1.out']," +
        " cmd='/bin/cp test/in $(location A1.out)')\n" +
        "genrule(name='A2', srcs=[], outs=['A2.out']," +
        " cmd='exit 42')\n" +
        "genrule(name='B', srcs=[], outs=['B.out']," +
        " cmd='/bin/cp test/in $(location B.out)')\n");
    write("test/in", "(input)");

    addOptions("--keep_going");
    build(true, GENRULE_ERROR, "//test:B", "//test:A");

    String stderr = recOutErr.errAsLatin1();
    assertThat(stderr).contains("Target //test:A failed to build\n");
    assertThat(stderr).containsMatch("Target //test:B up-to-date:\n" + "  .*/test/B\\.out\n");
  }

  @Test
  public void testNoKeepGoingResult() throws Exception {
    write("test/BUILD",
        "genrule(name='A', srcs=['A2'], outs=['A.out']," +
        " cmd='/bin/cp test/in $(location A.out)')\n" +
        "genrule(name='A2', srcs=[], outs=['A2.out']," +
        " cmd='exit 42')\n" +
        "genrule(name='B', srcs=[], outs=['B.out']," +
        " cmd='/bin/cp test/in $(location B.out)')\n");
    write("test/in", "(input)");

    build(true, GENRULE_ERROR, "//test:A", "//test:B");

    String stderr = recOutErr.errAsLatin1();
    assertThat(stderr).containsMatch("Target //test:A failed to build\n");
  }

  /**
   * This is a regression test for bug #823077, in which presence of artifacts
   * with null generating action caused targets always to be reported as failed.
   */
  @Test
  public void testKeepGoingResultWithNullActions() throws Exception {
    write("foo/BUILD",
          "sh_test(name='test_bar', srcs=['test_bar.sh'])\n" +
          "sh_test(name='test_foo', srcs=['test_foo.sh'])\n");
    write("foo/test_bar.sh", "echo bar is fine").setExecutable(true);

    addOptions("--keep_going");
    build(true, "no-error", "//foo:test_foo", "//foo:test_bar");

    String stderr = recOutErr.errAsLatin1();
    assertThat(stderr).contains("Target //foo:test_bar up-to-date:\n");
    assertThat(stderr).containsMatch("\n  .*/foo/test_bar\n");
    assertThat(stderr).doesNotContainMatch("\n  .*/foo/test_bar\\.sh\n");
    assertThat(stderr).contains("Target //foo:test_foo failed to build\n");
  }

  /**
   * This is a regression test for bug #1044470, in which targets with failed data dependencies
   * were still being reported as "up-to-date".
   */
  @Test
  public void testWithMissingData() throws Exception {
    write(
        "needsdata/BUILD",
        """
        cc_library(
            name = "needsdata",
            data = [":data_lib"],
        )

        cc_library(
            name = "data_lib",
            data = [":does_not_exist"],
        )
        """);

    // TODO(bazel-team): figure out why error message is non-deterministic with Skyframe full.
    // LOADING_AND_ANALYSIS loading_and_analysis cleanup.
    build(true, "input", "//needsdata");

    String stderr = recOutErr.errAsLatin1();
    assertThat(stderr).doesNotContain("Target //needsdata:needsdata up-to-date:\n");
    assertThat(stderr).contains("Target //needsdata:needsdata failed to build\n");
  }

  /**
   * This is a regression test for bug #987608, in which the temp artifacts (.ii and .s)
   * were not being reported as having been created.
   */
  @Test
  public void testWithSaveTemps() throws Exception {
    write("my_clib/BUILD",
        "cc_library(name='my_clib', srcs=['myclib.cc'])\n");
    write("my_clib/myclib.cc",
          "void f() {}");

    addOptions("--save_temps");
    build(false, "no-error", "//my_clib");

    String stderr = recOutErr.errAsLatin1();
    assertThat(stderr).contains("Target //my_clib:my_clib up-to-date:\n");
    assertThat(stderr).contains("blaze-bin/my_clib/_objs/my_clib/myclib.pic.s\n");
    assertThat(stderr).contains("blaze-bin/my_clib/_objs/my_clib/myclib.pic.ii\n");
  }

  /**
   * Test that changing the symlink prefix builds into another directory.
   */
  @Test
  public void testSymlinkPrefix() throws Exception {
    write("my_clib/BUILD",
        "cc_library(name='my_clib', srcs=['myclib.cc'])\n");
    write("my_clib/myclib.cc",
          "void f() {}");

    addOptions("--symlink_prefix=myblaze-");
    build(false, "no-error", "//my_clib");

    String stderr = recOutErr.errAsLatin1();
    assertThat(stderr).contains("Target //my_clib:my_clib up-to-date:\n");
    assertThat(stderr).contains("myblaze-bin/my_clib/libmy_clib.so\n");
    assertThat(stderr).contains("myblaze-bin/my_clib/libmy_clib.a\n");
  }

  /**
   * This is a regression test for bug #1013874, in which the temp artifacts (.ii and .s)
   * were not being reported as having been created within a failing target.
   * It's possible that one or more temps were successfully created, even if the
   * compilation failed.
   */
  @Test
  public void testFailedTargetWithSaveTemps() throws Exception {
    write("bad_clib/BUILD",
        "cc_library(name='bad_clib', srcs=['badlib.cc'])\n");
    // trigger a warning to make the build fail:
    //   "control reaches end of non-void function [-Werror,-Wreturn-type]"
    write("bad_clib/badlib.cc",
        "int f() { }");

    // We need to set --keep_going so that the temps get built even though the compilation fails.
    addOptions("--save_temps", "--keep_going");
    build(true, "compilation of rule '//bad_clib:bad_clib' failed", "//bad_clib");

    String stderr = recOutErr.errAsLatin1();
    assertThat(stderr).contains("Target //bad_clib:bad_clib failed to build");
    assertThat(stderr).contains("See temp at blaze-bin/bad_clib/_objs/bad_clib/badlib.pic.ii\n");
  }

  // Concrete implementations of this abstract test:

  /** Tests with 1 job. */
  @RunWith(JUnit4.class)
  public static class SequentialBuildResultTest extends BuildResultTestCase {
    @Override
    protected int numJobs() {
      return 1;
    }
  }

  /** Tests with 100 jobs. */
  @RunWith(JUnit4.class)
  public static class ParallelBuildResultTest extends BuildResultTestCase {
    @Override
    protected int numJobs() { return 100; }
  }
}
