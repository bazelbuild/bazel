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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ObjectArrays;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Before;
import org.junit.Test;

/** Test for {@link RunfilesSupport}. */
public abstract class AbstractRunfilesSupportTest extends BuildViewTestCase {

  protected abstract boolean useJdkLauncher();

  @Override
  protected final void useConfiguration(String... args) throws Exception {
    if (useJdkLauncher()) {
      super.useConfiguration(args);
    } else {
      super.useConfiguration(
          ObjectArrays.concat(args, "--java_launcher=//tools/java/launcher:run_java"));
    }
  }

  @Before
  public final void createDirectory() throws Exception {
    scratch.dir(outputBase.getParentDirectory() + "/blaze-bin");
  }

  @Test
  public void testWorkingDirectory() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        cc_test(
            name = "bar",
            srcs = ["bar.cc"],
        )
        """);
    ConfiguredTarget foo_bar;
    useConfiguration("--build_runfile_links");
    // we get expected runfiles directory
    foo_bar = getConfiguredTarget("//foo:bar");
    Path workDir1 = getRunfilesSupport(foo_bar).getRunfilesDirectory();
    assertThat(workDir1.asFragment().endsWith(PathFragment.create("foo/bar.runfiles"))).isTrue();

    // .. even when we change some options
    useConfiguration("--nobuild_runfile_links");
    // Reconfigured targets.
    foo_bar = getConfiguredTarget("//foo:bar");
    Path workDir2 = getRunfilesSupport(foo_bar).getRunfilesDirectory();
    assertThat(workDir2).isEqualTo(workDir1);
  }

  @Test
  public void testVisitingPackageGroups() throws Exception {
    scratch.file("honeydew/BUILD", "package_group(name='honeydew')");

    collectRunfiles(getConfiguredTarget("//honeydew"));
  }
}
