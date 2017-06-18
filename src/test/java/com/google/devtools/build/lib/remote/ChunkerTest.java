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
import com.google.devtools.build.lib.remote.Chunker.Chunk;
import com.google.devtools.remoteexecution.v1test.Digest;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Chunker}. */
@RunWith(JUnit4.class)
public class ChunkerTest {

  @Test
  public void testChunker() throws Exception {
    byte[] b1 = "abcdefg".getBytes(UTF_8);
    byte[] b2 = "hij".getBytes(UTF_8);
    byte[] b3 = "klmnopqrstuvwxyz".getBytes(UTF_8);
    Digest d1 = Digests.computeDigest(b1);
    Digest d2 = Digests.computeDigest(b2);
    Digest d3 = Digests.computeDigest(b3);
    Chunker c = new Chunker.Builder().chunkSize(5).addInput(b1).addInput(b2).addInput(b3).build();
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(new Chunk(d1, "abcde".getBytes(UTF_8), 0));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(new Chunk(d1, "fg".getBytes(UTF_8), 5));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(new Chunk(d2, "hij".getBytes(UTF_8), 0));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(new Chunk(d3, "klmno".getBytes(UTF_8), 0));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(new Chunk(d3, "pqrst".getBytes(UTF_8), 5));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(new Chunk(d3, "uvwxy".getBytes(UTF_8), 10));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(new Chunk(d3, "z".getBytes(UTF_8), 15));
    assertThat(c.hasNext()).isFalse();
  }

  @Test
  public void testIgnoresUnmentionedDigests() throws Exception {
    byte[] b1 = "a".getBytes(UTF_8);
    byte[] b2 = "bb".getBytes(UTF_8);
    byte[] b3 = "ccc".getBytes(UTF_8);
    byte[] b4 = "dddd".getBytes(UTF_8);
    Digest d1 = Digests.computeDigest(b1);
    Digest d3 = Digests.computeDigest(b3);
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
    assertThat(c.next()).isEqualTo(new Chunk(d1, "a".getBytes(UTF_8), 0));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(new Chunk(d3, "cc".getBytes(UTF_8), 0));
    assertThat(c.hasNext()).isTrue();
    assertThat(c.next()).isEqualTo(new Chunk(d3, "c".getBytes(UTF_8), 2));
    assertThat(c.hasNext()).isFalse();
  }
}
