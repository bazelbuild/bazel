// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.google.common.collect.ImmutableList;
import com.google.common.jimfs.Jimfs;
import com.google.common.truth.Truth;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link UnvalidatedAndroidDirectories}. */
@RunWith(JUnit4.class)
public class UnvalidatedAndroidDirectoriesTest {
  private FileSystem fileSystem;
  private Path res;
  private Path otherRes;
  private Path assets;
  private Path otherAssets;

  @Before public void setUp() throws Exception {
    fileSystem = Jimfs.newFileSystem();
    Path root = fileSystem.getPath("");
    res = Files.createDirectories(root.resolve("res"));
    otherRes = Files.createDirectories(root.resolve("otherres"));
    assets = Files.createDirectories(root.resolve("assets"));
    otherAssets = Files.createDirectories(root.resolve("otherassets"));
  }

  @Test public void flagFullParse() throws Exception {
    Truth.assertThat(
        UnvalidatedAndroidDirectories.valueOf(
            "res#otherres:assets#otherassets", fileSystem)
    ).isEqualTo(
        new UnvalidatedAndroidDirectories(ImmutableList.of(res, otherRes),
            ImmutableList.of(assets, otherAssets)));
  }

  @Test public void flagParseWithEmptyAssets() throws Exception {
    Truth.assertThat(
        UnvalidatedAndroidDirectories.valueOf(
            "res:", fileSystem)
    ).isEqualTo(
        new UnvalidatedAndroidDirectories(ImmutableList.of(res),
            ImmutableList.<Path>of()));
  }

  @Test public void flagParseWithEmptyResources() throws Exception {
    Truth.assertThat(
        UnvalidatedAndroidDirectories.valueOf(
            ":assets", fileSystem)
    ).isEqualTo(
        new UnvalidatedAndroidDirectories(ImmutableList.<Path>of(),
            ImmutableList.of(assets)));
  }

  @Test public void flagParseWithEmptyResourcesAndAssets() throws Exception {
    Truth.assertThat(
        UnvalidatedAndroidDirectories.valueOf(
            ":", fileSystem)
    ).isEqualTo(
        new UnvalidatedAndroidDirectories(ImmutableList.<Path>of(),
            ImmutableList.<Path>of()));
  }

}
