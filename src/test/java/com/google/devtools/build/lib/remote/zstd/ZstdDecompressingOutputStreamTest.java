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

import com.github.luben.zstd.Zstd;
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
    ZstdDecompressingOutputStream zdos = new ZstdDecompressingOutputStream(baos);
    zdos.write(compressed);
    zdos.flush();

    assertThat(baos.toByteArray()).isEqualTo(data);
  }
}
