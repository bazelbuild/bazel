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
package com.google.devtools.build.android.dexer;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.android.dex.Dex;
import com.android.dx.dex.code.PositionList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.runfiles.Runfiles;
import java.nio.file.FileSystems;
import java.nio.file.Paths;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DexBuilder}. */
@RunWith(JUnit4.class)
public class DexBuilderTest {

  @Test
  public void testBuildDexArchive() throws Exception {
    DexBuilder.Options options = new DexBuilder.Options();
    // Use Jar file that has this test in it as the input Jar
    Runfiles runfiles = Runfiles.create();
    options.inputJar = Paths.get(runfiles.rlocation(System.getProperty("testinputjar")));
    options.outputZip =
        FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR"), "dex_builder_test.zip");
    options.maxThreads = 1;
    Dexing.DexingOptions dexingOptions = new Dexing.DexingOptions();
    dexingOptions.optimize = true;
    dexingOptions.positionInfo = PositionList.LINES;
    DexBuilder.buildDexArchive(options, new Dexing(dexingOptions));
    assertThat(options.outputZip.toFile().exists()).isTrue();

    HashSet<String> files = new HashSet<>();
    try (ZipFile zip = new ZipFile(options.outputZip.toFile())) {
      Enumeration<? extends ZipEntry> entries = zip.entries();
      while (entries.hasMoreElements()) {
        ZipEntry entry = entries.nextElement();
        files.add(entry.getName());
        if (entry.getName().endsWith(".dex")) {
          Dex dex = new Dex(zip.getInputStream(entry));
          assertThat(dex.classDefs()).named(entry.getName()).hasSize(1);
        } else if (entry.getName().endsWith("/testresource.txt")) {
          byte[] content = ByteStreams.toByteArray(zip.getInputStream(entry));
          assertThat(content).named(entry.getName()).isEqualTo("test".getBytes(UTF_8));
        }
      }
    }
    // Make sure this test is in the Zip file, which also means we parsed its dex code above
    assertThat(files).contains(getClass().getName().replace('.', '/') + ".class.dex");
    // Make sure test resource is in the Zip file, which also means it had the expected content
    assertThat(files)
        .contains(getClass().getPackage().getName().replace('.', '/') + "/testresource.txt");
  }
}

