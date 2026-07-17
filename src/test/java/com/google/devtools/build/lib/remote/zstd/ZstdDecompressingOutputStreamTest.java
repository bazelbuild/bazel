// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.zstd;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.github.luben.zstd.Zstd;
import com.github.luben.zstd.ZstdOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ZstdDecompressingOutputStream}. */
@RunWith(JUnit4.class)
public class ZstdDecompressingOutputStreamTest {
  @Test
  public void decompressionWorks() throws IOException {
    Random rand = new Random();
    byte[] data = new byte[50];
    rand.nextBytes(data);
    byte[] compressed = Zstd.compress(data);

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (ZstdDecompressingOutputStream zdos = new ZstdDecompressingOutputStream(baos)) {
      zdos.write(compressed);
      zdos.flush();
    }

    assertThat(baos.toByteArray()).isEqualTo(data);
  }

  @Test
  public void streamCanBeDecompressedOneByteAtATime() throws IOException {
    Random rand = new Random();
    byte[] data = new byte[50];
    rand.nextBytes(data);
    byte[] compressed = Zstd.compress(data);

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (ZstdDecompressingOutputStream zdos = new ZstdDecompressingOutputStream(baos)) {
      for (byte b : compressed) {
        zdos.write(b);
      }
      zdos.flush();
    }

    assertThat(baos.toByteArray()).isEqualTo(data);
  }

  @Test
  public void bytesWrittenMatchesDecompressedBytes() throws IOException {
    byte[] data = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".getBytes(UTF_8);

    ByteArrayOutputStream compressed = new ByteArrayOutputStream();
    try (ZstdOutputStream zos = new ZstdOutputStream(compressed)) {
      zos.setCloseFrameOnFlush(true);
      for (int i = 0; i < data.length; i++) {
        zos.write(data[i]);
        if (i % 5 == 0) {
          // Create multiple frames of 5 bytes each.
          zos.flush();
        }
      }
    }

    ByteArrayOutputStream decompressed = new ByteArrayOutputStream();
    try (ZstdDecompressingOutputStream zdos = new ZstdDecompressingOutputStream(decompressed)) {
      for (byte b : compressed.toByteArray()) {
        zdos.write(b);
        zdos.flush();
      }
    }
    assertThat(decompressed.toByteArray()).isEqualTo(data);
  }
}
