// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.remote.RemoteProtocol.BlobChunk;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.protobuf.ByteString;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Chunker}. */
@RunWith(JUnit4.class)
public class ChunkerTest {

  static BlobChunk buildChunk(long offset, String data) {
    return BlobChunk.newBuilder().setOffset(offset).setData(ByteString.copyFromUtf8(data)).build();
  }

  static BlobChunk buildChunk(ContentDigest digest, String data) {
    return BlobChunk.newBuilder().setDigest(digest).setData(ByteString.copyFromUtf8(data)).build();
  }

  @Test
  public void testChunker() throws Exception {
    byte[] b1 = "abcdefg".getBytes(UTF_8);
    byte[] b2 = "hij".getBytes(UTF_8);
    byte[] b3 = "klmnopqrstuvwxyz".getBytes(UTF_8);
    ContentDigest d1 = ContentDigests.computeDigest(b1);
    ContentDigest d2 = ContentDigests.computeDigest(b2);
    ContentDigest d3 = ContentDigests.computeDigest(b3);
    Chunker c = new Chunker.Builder().chunkSize(5).addInput(b1).addInput(b2).addInput(b3).build();
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(buildChunk(d1, "abcde"));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(buildChunk(5, "fg"));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(buildChunk(d2, "hij"));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(buildChunk(d3, "klmno"));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(buildChunk(5, "pqrst"));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(buildChunk(10, "uvwxy"));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(buildChunk(15, "z"));
    assertThat(c.hasNext()).isFalse();
  }

  @Test
  public void testIgnoresUnmentionedDigests() throws Exception {
    byte[] b1 = "a".getBytes(UTF_8);
    byte[] b2 = "bb".getBytes(UTF_8);
    byte[] b3 = "ccc".getBytes(UTF_8);
    byte[] b4 = "dddd".getBytes(UTF_8);
    ContentDigest d1 = ContentDigests.computeDigest(b1);
    ContentDigest d2 = ContentDigests.computeDigest(b2);
    ContentDigest d3 = ContentDigests.computeDigest(b3);
    ContentDigest d4 = ContentDigests.computeDigest(b4);
    Chunker c =
        new Chunker.Builder()
            .chunkSize(2)
            .onlyUseDigests(ImmutableSet.of(d1, d3))
            .addInput(b1)
            .addInput(b2)
            .addInput(b3)
            .addInput(b4)
            .build();
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(buildChunk(d1, "a"));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(buildChunk(d3, "cc"));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(buildChunk(2, "c"));
    assertThat(c.hasNext()).isFalse();
  }
}
