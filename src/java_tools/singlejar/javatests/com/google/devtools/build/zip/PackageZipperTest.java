// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.zip;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PackageZipper}. */
@RunWith(JUnit4.class)
public class PackageZipperTest {

  /** 1980-01-01 00:00:00 packed into the DOS date and time fields. */
  private static final int DOS_1980_01_01 = 0x00210000;

  /** Offset of the modification time field in a local file header. */
  private static final int LOCAL_HEADER_MOD_TIME_OFFSET = 10;

  private Path tmpDir;
  private Path outputZip;
  private Path serverJar;
  private Path installBaseKey;

  @Before
  public void setUpInputs() throws IOException {
    tmpDir = Files.createTempDirectory("package_zipper_test");
    outputZip = tmpDir.resolve("package.zip");
    serverJar = tmpDir.resolve("A-server.jar");
    installBaseKey = tmpDir.resolve("install_base_key");

    try (OutputStream out = Files.newOutputStream(serverJar);
        ZipOutputStream jar = new ZipOutputStream(out)) {
      jar.putNextEntry(new ZipEntry("build-data.properties"));
      jar.write("build.label=1.2.3\n".getBytes(UTF_8));
      jar.closeEntry();
    }
    Files.write(installBaseKey, "0123456789abcdef".getBytes(UTF_8));
  }

  @Test
  public void writesEntriesWithLocalDosEpochTimestamp() throws IOException {
    Path file = Files.write(tmpDir.resolve("foo.txt"), "foo".getBytes(UTF_8));

    PackageZipper.main(
        new String[] {
          outputZip.toString(), serverJar.toString(), installBaseKey.toString(), file.toString()
        });

    try (ZipFile zip = new ZipFile(outputZip.toFile(), UTF_8)) {
      List<String> names = new ArrayList<>();
      for (ZipEntry entry : Collections.list(zip.entries())) {
        names.add(entry.getName());
        // The recorded date is the same in every time zone, so the archive bytes are as well.
        assertThat(ZipUtil.unixToDosTime(entry.getTime())).isEqualTo(DOS_1980_01_01);
      }
      assertThat(names)
          .containsExactly("A-server.jar", "build-label.txt", "foo.txt", "install_base_key");
    }

    // The first local file header starts at offset 0.
    byte[] archive = Files.readAllBytes(outputZip);
    assertThat(ZipUtil.get32(archive, LOCAL_HEADER_MOD_TIME_OFFSET)).isEqualTo(DOS_1980_01_01);
  }
}
