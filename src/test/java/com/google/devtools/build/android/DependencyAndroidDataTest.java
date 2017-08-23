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
import com.google.devtools.build.android.aapt2.StaticLibrary;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the {@link DependencyAndroidData}.
 */
@RunWith(JUnit4.class)
public class DependencyAndroidDataTest {
  private FileSystem fileSystem;
  private Path root;
  private Path rTxt;
  private Path manifest;
  private Path res;
  private Path otherRes;
  private Path assets;
  private Path otherAssets;
  private Path symbols;
  private StaticLibrary staticLibrary;

  @Before public void setUp() throws Exception {
    fileSystem = Jimfs.newFileSystem();
    root = Files.createDirectories(fileSystem.getPath(""));
    rTxt = Files.createFile(root.resolve("r.txt"));
    staticLibrary = StaticLibrary.from(Files.createFile(root.resolve("static.library.ap_")));
    symbols = Files.createFile(root.resolve("symbols.bin"));
    manifest = Files.createFile(root.resolve("AndroidManifest.xml"));
    res = Files.createDirectories(root.resolve("res"));
    otherRes = Files.createDirectories(root.resolve("otherres"));
    assets = Files.createDirectories(root.resolve("assets"));
    otherAssets = Files.createDirectories(root.resolve("otherassets"));
  }

  @Test public void flagFullParse() throws Exception{
    Truth.assertThat(
            DependencyAndroidData.valueOf(
                "res#otherres:assets#otherassets:AndroidManifest.xml:r.txt:symbols.bin:static.library.ap_",
                fileSystem))
        .isEqualTo(
            new DependencyAndroidData(
                ImmutableList.of(res, otherRes),
                ImmutableList.of(assets, otherAssets),
                manifest,
                rTxt,
                symbols,
                staticLibrary));
  }

  @Test public void flagParseWithNoSymbolsFile() throws Exception{
    Truth.assertThat(
            DependencyAndroidData.valueOf(
                "res#otherres:assets#otherassets:AndroidManifest.xml:r.txt:", fileSystem))
        .isEqualTo(
            new DependencyAndroidData(
                ImmutableList.of(res, otherRes),
                ImmutableList.of(assets, otherAssets),
                manifest,
                rTxt,
                null,
                null));
  }

  @Test public void flagParseOmittedSymbolsFile() throws Exception{
    Truth.assertThat(
            DependencyAndroidData.valueOf(
                "res#otherres:assets#otherassets:AndroidManifest.xml:r.txt", fileSystem))
        .isEqualTo(
            new DependencyAndroidData(
                ImmutableList.of(res, otherRes),
                ImmutableList.of(assets, otherAssets),
                manifest,
                rTxt,
                null,
                null));
  }

  @Test public void flagParseWithEmptyResources() throws Exception{
    Truth.assertThat(
            DependencyAndroidData.valueOf(
                ":assets:AndroidManifest.xml:r.txt:symbols.bin", fileSystem))
        .isEqualTo(
            new DependencyAndroidData(
                ImmutableList.<Path>of(), ImmutableList.of(assets), manifest, rTxt, symbols, null));
  }

  @Test public void flagParseWithEmptyAssets() throws Exception{
    Truth.assertThat(
            DependencyAndroidData.valueOf("res::AndroidManifest.xml:r.txt:symbols.bin", fileSystem))
        .isEqualTo(
            new DependencyAndroidData(
                ImmutableList.of(res), ImmutableList.<Path>of(), manifest, rTxt, symbols, null));
  }

  @Test public void flagParseWithEmptyResourcesAndAssets() throws Exception{
    Truth.assertThat(
            DependencyAndroidData.valueOf("::AndroidManifest.xml:r.txt:symbols.bin", fileSystem))
        .isEqualTo(
            new DependencyAndroidData(
                ImmutableList.<Path>of(), ImmutableList.<Path>of(), manifest, rTxt, symbols, null));
  }

  @Test public void flagNoManifestFails() {
    try {
      DependencyAndroidData.valueOf(":::r.txt", fileSystem);
      fail("expected exception for bad flag format");
    } catch (IllegalArgumentException expected) {
    }
  }

  @Test public void flagMissingManifestFails() {
    try {
      DependencyAndroidData.valueOf("::Manifest.xml:r.txt:symbols.bin", fileSystem);
      fail("expected exception for bad flag format");
    } catch (IllegalArgumentException expected) {
    }
  }

  @Test public void flagNoRTxtFails() {
    try {
      DependencyAndroidData.valueOf("::AndroidManifest.xml:", fileSystem);
      fail("expected exception for bad flag format");
    } catch (IllegalArgumentException expected) {
    }
  }

  @Test public void flagMissingRTxtFails() {
    try {
      DependencyAndroidData.valueOf("::Manifest.xml:missing_file", fileSystem);
      fail("expected exception for bad flag format");
    } catch (IllegalArgumentException expected) {
    }
  }

  @Test public void flagMissingSymbolsFails() {
    try {
      DependencyAndroidData.valueOf("::Manifest.xml:r.txt:missing_file", fileSystem);
      fail("expected exception for bad flag format");
    } catch (IllegalArgumentException expected) {
    }
  }
}
