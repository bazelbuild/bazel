// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.skyframe.SkyfocusState.Request;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Skyfocus integration tests */
@RunWith(JUnit4.class)
public final class SkyfocusIntegrationTest extends BuildIntegrationTestCase {

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();
    addOptions("--experimental_enable_skyfocus");
  }

  @Test
  public void workingSet_canBeUsedWithBuildCommand() throws Exception {
    addOptions("--experimental_working_set=hello/x.txt", "--experimental_skyfocus_dump_keys=count");
    write("hello/x.txt", "x");
    write(
        "hello/BUILD",
        """
        genrule(
            name = "target",
            srcs = ["x.txt"],
            outs = ["out"],
            cmd = "cat $< > $@",
        )
        """);

    buildTarget("//hello/...");
    assertContainsEvent("Updated working set successfully.");
    assertContainsEvent("Focusing on");
    assertContainsEvent("Node count:");

    assertThat(getSkyframeExecutor().getSkyfocusState().workingSet()).hasSize(1);
  }

  @Test
  public void workingSet_withFiles_correctlyRebuilds() throws Exception {
    addOptions("--experimental_working_set=hello/x.txt");
    write("hello/x.txt", "x");
    write("hello/y.txt", "y");
    write(
        "hello/BUILD",
        """
        genrule(
            name = "target",
            srcs = ["x.txt", "y.txt"],
            outs = ["out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);

    buildTarget("//hello/...");
    assertContents("x\ny", "//hello:target");

    write("hello/x.txt", "x2");
    buildTarget("//hello/...");
    assertContents("x2\ny", "//hello:target");
  }

  @Test
  public void workingSet_withDirs_correctlyRebuilds() throws Exception {
    /*
     * Setting directories in the working set works, because the rdep edges look like:
     *
     * FILE_STATE:[dir] -> FILE:[dir] -> FILE:[dir/BUILD], FILE:[dir/file.txt]
     *
     * ...and the FILE SkyKeys directly depend on their respective FILE_STATE SkyKeys,
     * which are the nodes that are invalidated by SkyframeExecutor#handleDiffs
     * at the start of every build, and are also kept by Skyfocus.
     *
     * In other words, defining a working set of directories will automatically
     * include all the files under those directories for focusing.
     */

    // Define working set to be a directory, not file
    addOptions("--experimental_working_set=hello");
    write("hello/x.txt", "x");
    write("hello/y.txt", "y");
    write(
        "hello/BUILD",
        """
        genrule(
            name = "target",
            srcs = ["x.txt", "y.txt"],
            outs = ["out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);

    buildTarget("//hello/...");
    assertContents("x\ny", "//hello:target");

    write("hello/x.txt", "x2");
    buildTarget("//hello/...");
    // Correctly rebuilds referenced source file
    assertContents("x2\ny", "//hello:target");

    write(
        "hello/BUILD",
        """
        genrule(
            name = "y_only",
            srcs = ["y.txt"],
            outs = ["out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    buildTarget("//hello/...");
    // Correctly reanalyzes BUILD file
    assertContents("y", "//hello:y_only");
  }

  @Test
  public void workingSet_nestedDirs_correctlyRebuilds() throws Exception {
    // Define working set to be the parent package
    addOptions("--experimental_working_set=hello");
    write("hello/x.txt", "x");
    write(
        "hello/BUILD",
        """
        genrule(
            name = "target",
            srcs = ["x.txt", "//hello/world:target"],
            outs = ["out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);
    write("hello/world/y.txt", "y");
    write(
        "hello/world/BUILD",
        """
        genrule(
            name = "target",
            srcs = ["y.txt"],
            outs = ["out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);

    buildTarget("//hello/...");
    assertContents("x\ny", "//hello:target");

    write("hello/x.txt", "x2");
    buildTarget("//hello/...");
    // Rebuilds when parent package's source file changes
    assertContents("x2\ny", "//hello:target");

    write("hello/world/y.txt", "y2");
    buildTarget("//hello/...");
    // Rebuilds when child package's source file changes
    assertContents("x2\ny2", "//hello:target");
  }

  @Test
  public void newNonFocusedTargets_canBeBuilt() throws Exception {
    addOptions("--experimental_working_set=hello/x.txt");
    write("hello/x.txt", "x");
    write(
        "hello/BUILD",
        """
        genrule(
            name = "g",
            srcs = ["x.txt"],
            outs = ["g.txt"],
            cmd = "cat $(SRCS) > $@",
        )
        genrule(
            name = "g2",
            srcs = ["x.txt"],
            outs = ["g2.txt"],
            cmd = "cat $(SRCS) > $@",
        )
        genrule(
            name = "g3",
            outs = ["g3.txt"],
            cmd = "touch $@",
        )
        """);
    buildTarget("//hello:g");
    write("hello/x.txt", "x2");
    buildTarget("//hello:g");
    buildTarget("//hello:g2");
    buildTarget("//hello:g3");
  }

  @Test
  public void skyfocus_doesNotRun_forUnsuccessfulBuilds() throws Exception {
    addOptions("--experimental_working_set=hello/x.txt");
    write("hello/x.txt", "x");
    write(
        "hello/BUILD",
        """
        genrule(
          # error
        )
        """);
    TargetParsingException e =
        assertThrows(TargetParsingException.class, () -> buildTarget("//hello/..."));

    assertThat(e).hasMessageThat().contains("Package 'hello' contains errors");

    assertThat(getSkyframeExecutor().getSkyfocusState().enabled()).isTrue();
    assertThat(getSkyframeExecutor().getSkyfocusState().request()).isNotEqualTo(Request.DO_NOTHING);
    assertThat(getSkyframeExecutor().getSkyfocusState().verificationSet()).isEmpty();
  }

  @Test
  public void workingSet_reduced_withoutReanalysis() throws Exception {
    addOptions("--experimental_working_set=hello/x.txt,hello/y.txt");
    write("hello/x.txt", "x");
    write("hello/y.txt", "y");
    write(
        "hello/BUILD",
        """
        genrule(
            name = "target",
            srcs = ["x.txt", "y.txt"],
            outs = ["out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);

    buildTarget("//hello/...");
    assertThat(getSkyframeExecutor().getSkyfocusState().workingSet()).hasSize(2);

    resetOptions();
    setupOptions();
    addOptions("--experimental_working_set=hello/x.txt");
    write("hello/x.txt", "x2");

    buildTarget("//hello/...");
    assertDoesNotContainEvent("discarding analysis cache");
    assertContents("x2\ny", "//hello:target");
    assertThat(getSkyframeExecutor().getSkyfocusState().workingSet()).hasSize(1);
  }

  @Test
  public void workingSet_expanded_withReanalysis() throws Exception {
    addOptions("--experimental_working_set=hello/x.txt");
    write("hello/x.txt", "x");
    write("hello/y.txt", "y");
    write(
        "hello/BUILD",
        """
        genrule(
            name = "target",
            srcs = ["x.txt", "y.txt"],
            outs = ["out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);

    buildTarget("//hello/...");
    assertThat(getSkyframeExecutor().getSkyfocusState().workingSet()).hasSize(1);

    resetOptions();
    setupOptions();
    addOptions("--experimental_working_set=hello/x.txt,hello/y.txt");
    write("hello/x.txt", "x2");

    buildTarget("//hello/...");
    assertContainsEvent("discarding analysis cache");
    assertContents("x2\ny", "//hello:target");
    assertThat(getSkyframeExecutor().getSkyfocusState().workingSet()).hasSize(2);
  }
}
