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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.zip.CRC32;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link AdjustSfx}. */
@RunWith(JUnit4.class)
public class AdjustSfxTest {

  @Rule public TemporaryFolder tmp = new TemporaryFolder();

  @Test
  public void testAdjustSfx() throws IOException {
    File zipFile = tmp.newFile("valid.zip");
    byte[] content = "hello".getBytes(UTF_8);
    CRC32 crc = new CRC32();
    crc.update(content);
    try (ZipWriter writer = new ZipWriter(new FileOutputStream(zipFile), UTF_8, false)) {
      ZipFileEntry entry = new ZipFileEntry("foo.txt");
      entry.setTime(ZipUtil.DOS_EPOCH);
      entry.setCrc(crc.getValue());
      entry.setSize(content.length);
      entry.setCompressedSize(content.length);
      writer.putNextEntry(entry);
      writer.write(content);
    }

    byte[] zipBytes = Files.readAllBytes(zipFile.toPath());
    int preambleLength = 5;
    byte[] sfxBytes = new byte[preambleLength + zipBytes.length];
    System.arraycopy(zipBytes, 0, sfxBytes, preambleLength, zipBytes.length);

    File sfxFile = tmp.newFile("sfx.exe");
    Files.write(sfxFile.toPath(), sfxBytes);

    File adjustedFile = tmp.newFile("adjusted.exe");
    AdjustSfx.main(new String[] {sfxFile.getAbsolutePath(), adjustedFile.getAbsolutePath()});

    try (ZipReader reader = new ZipReader(adjustedFile, UTF_8)) {
      assertThat(reader.entries()).hasSize(1);
      ZipFileEntry entry = reader.getEntry("foo.txt");
      assertThat(entry).isNotNull();
      assertThat(entry.getCrc()).isEqualTo(crc.getValue());
      assertThat(entry.getSize()).isEqualTo(content.length);
      try (InputStream in = reader.getInputStream(entry)) {
        byte[] readContent = in.readAllBytes();
        assertThat(readContent).isEqualTo(content);
      }
    }
  }
}
