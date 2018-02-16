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

import static org.junit.Assert.fail;

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

/** Tests for {@link UnvalidatedAndroidData}. */
@RunWith(JUnit4.class)
public class UnvalidatedAndroidDataTest {
  private FileSystem fileSystem;
  private Path root;
  private Path manifest;
  private Path res;
  private Path otherRes;
  private Path assets;
  private Path otherAssets;

  @Before public void setUp() throws Exception {
    fileSystem = Jimfs.newFileSystem();
    root = fileSystem.getPath("");
    manifest = Files.createFile(root.resolve("AndroidManifest.xml"));
    res = Files.createDirectories(root.resolve("res"));
    otherRes = Files.createDirectories(root.resolve("otherres"));
    assets = Files.createDirectories(root.resolve("assets"));
    otherAssets = Files.createDirectories(root.resolve("otherassets"));
  }

  @Test public void flagFullParse() throws Exception {
    Truth.assertThat(
        UnvalidatedAndroidData.valueOf(
            "res#otherres:assets#otherassets:AndroidManifest.xml", fileSystem)
        ).isEqualTo(
            new UnvalidatedAndroidData(ImmutableList.of(res, otherRes),
                ImmutableList.of(assets, otherAssets),
                manifest));
  }

  @Test public void flagParseWithEmptyResources() throws Exception {
    Truth.assertThat(
        UnvalidatedAndroidData.valueOf(
            ":assets:AndroidManifest.xml", fileSystem)
        ).isEqualTo(
            new UnvalidatedAndroidData(ImmutableList.<Path>of(),
                ImmutableList.of(assets),
                manifest));
  }

  @Test public void flagParseWithEmptyAssets() throws Exception {
    Truth.assertThat(
        UnvalidatedAndroidData.valueOf(
            "res::AndroidManifest.xml", fileSystem)
        ).isEqualTo(
            new UnvalidatedAndroidData(ImmutableList.of(res),
                ImmutableList.<Path>of(),
                manifest));
  }

  @Test public void flagParseWithEmptyResourcesAndAssets() throws Exception {
    Truth.assertThat(
        UnvalidatedAndroidData.valueOf(
            "::AndroidManifest.xml", fileSystem)
        ).isEqualTo(
            new UnvalidatedAndroidData(ImmutableList.<Path>of(),
                ImmutableList.<Path>of(),
                manifest));
  }

  @Test public void flagNoManifestFails() {
    try {
      UnvalidatedAndroidData.valueOf("::", fileSystem);
      fail("expected exception for bad flag format");
    } catch (IllegalArgumentException expected) {
    }
  }

  @Test public void flagMissingManifestFails() {
    try {
      UnvalidatedAndroidData.valueOf("::Manifest.xml", fileSystem);
      fail("expected exception for bad flag format");
    } catch (IllegalArgumentException expected) {
    }
  }
}
