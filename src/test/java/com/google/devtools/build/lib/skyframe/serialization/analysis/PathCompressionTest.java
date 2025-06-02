// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.analysis.PathCompression.Compressor;
import com.google.devtools.build.lib.skyframe.serialization.analysis.PathCompression.Decompressor;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class PathCompressionTest {
  private static final ImmutableList<String> PATHS =
      ImmutableList.of("a/b/c/d", "a/c/d", "a/c/d/e", "a/c", "b/c", "b/c", "");

  private byte[] byteArrayOutput;
  private CodedInputStream compressedPathsIn;

  @Before
  public void setup() throws IOException {
    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    CodedOutputStream compressedPathsOut = CodedOutputStream.newInstance(byteArrayOutputStream);
    compressedPathsOut.writeInt32NoTag(0);
    compressedPathsOut.writeStringNoTag("a/b/c/d");
    compressedPathsOut.writeInt32NoTag(2);
    compressedPathsOut.writeStringNoTag("c/d");
    compressedPathsOut.writeInt32NoTag(5);
    compressedPathsOut.writeStringNoTag("/e");
    compressedPathsOut.writeInt32NoTag(3);
    compressedPathsOut.writeStringNoTag("");
    compressedPathsOut.writeInt32NoTag(0);
    compressedPathsOut.writeStringNoTag("b/c");
    compressedPathsOut.writeInt32NoTag(3);
    compressedPathsOut.writeStringNoTag("");
    compressedPathsOut.writeInt32NoTag(0);
    compressedPathsOut.writeStringNoTag("");
    compressedPathsOut.flush();

    byteArrayOutput = byteArrayOutputStream.toByteArray();
    compressedPathsIn = CodedInputStream.newInstance(byteArrayOutput);
  }

  @Test
  public void testCompressor() throws Exception {
    ByteArrayOutputStream compressorByteArray = new ByteArrayOutputStream();
    CodedOutputStream out = CodedOutputStream.newInstance(compressorByteArray);
    Compressor compressor = PathCompression.compressor();
    for (String path : PATHS) {
      compressor.compressTo(path, out);
    }
    out.flush();
    assertThat(compressorByteArray.toByteArray()).isEqualTo(byteArrayOutput);
  }

  @Test
  public void testDecompressor() throws Exception {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    Decompressor decompressor = PathCompression.decompressor();
    for (int i = 0; i < PATHS.size(); i++) {
      result.add(decompressor.decompressFrom(compressedPathsIn));
    }
    assertThat(result.build()).isEqualTo(PATHS);
  }
}
