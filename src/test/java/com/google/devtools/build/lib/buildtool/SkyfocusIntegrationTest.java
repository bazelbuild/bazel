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

import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
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
}
