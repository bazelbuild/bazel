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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class PathPrettyPrinterTest {

  @Test
  public void getPrettyPath_pathUnderSymlinkTarget_returnsPathUnderConvenienceLink() {
    PathPrettyPrinter underTest =
        new PathPrettyPrinter(
            /* symlinkPrefix= */ "ignored",
            ImmutableMap.of(
                PathFragment.create("not-blaze-out"),
                PathFragment.create("/output/execroot/not-stuff"),
                PathFragment.create("blaze-out"),
                PathFragment.create("/output/execroot/stuff")));

    PathFragment path = PathFragment.create("/output/execroot/stuff/really");
    assertThat(underTest.getPrettyPath(path)).isEqualTo(PathFragment.create("blaze-out/really"));
  }

  @Test
  public void getPrettyPath_pathNotUnderSymlinkTarget_returnsOriginalPath() {
    PathPrettyPrinter underTest =
        new PathPrettyPrinter(
            /* symlinkPrefix= */ "ignored",
            ImmutableMap.of(
                PathFragment.create("blaze-out"), PathFragment.create("/output/execroot/stuff")));

    PathFragment path = PathFragment.create("/output/execroot/not-stuff/really");
    assertThat(underTest.getPrettyPath(path)).isEqualTo(path);
  }

  @Test
  public void getPrettyPath_noCreateSymlinksPrefix_returnsOriginalPath() {
    PathPrettyPrinter underTest =
        new PathPrettyPrinter(
            /* symlinkPrefix= */ "/",
            ImmutableMap.of(
                PathFragment.create("blaze-out"), PathFragment.create("/output/execroot/stuff")));

    PathFragment path = PathFragment.create("/output/execroot/stuff/really");
    assertThat(underTest.getPrettyPath(path)).isEqualTo(path);
  }
}
