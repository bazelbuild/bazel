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
import static junit.framework.TestCase.fail;

import com.google.devtools.build.lib.remote.Chunker.Chunk;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.protobuf.ByteString;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Chunker}. */
@RunWith(JUnit4.class)
public class ChunkerTest {

  @Test
  public void chunkingShouldWork() throws IOException {
    Random rand = new Random();
    byte[] expectedData = new byte[21];
    rand.nextBytes(expectedData);
    Digest expectedDigest = Digests.computeDigest(expectedData);

    Chunker chunker = new Chunker(expectedData, 10);

    ByteArrayOutputStream actualData = new ByteArrayOutputStream();

    assertThat(chunker.hasNext()).isTrue();
    Chunk next = chunker.next();
    assertThat(next.getOffset()).isEqualTo(0);
    assertThat(next.getData()).hasSize(10);
    assertThat(next.getDigest()).isEqualTo(expectedDigest);
    next.getData().writeTo(actualData);

    assertThat(chunker.hasNext()).isTrue();
    next = chunker.next();
    assertThat(next.getOffset()).isEqualTo(10);
    assertThat(next.getData()).hasSize(10);
    assertThat(next.getDigest()).isEqualTo(expectedDigest);
    next.getData().writeTo(actualData);

    assertThat(chunker.hasNext()).isTrue();
    next = chunker.next();
    assertThat(next.getOffset()).isEqualTo(20);
    assertThat(next.getData()).hasSize(1);
    assertThat(next.getDigest()).isEqualTo(expectedDigest);
    next.getData().writeTo(actualData);

    assertThat(chunker.hasNext()).isFalse();

    assertThat(expectedData).isEqualTo(actualData.toByteArray());
  }

  @Test
  public void nextShouldThrowIfNoMoreData() throws IOException {
    byte[] data = new byte[10];
    Chunker chunker = new Chunker(data, 10);

    assertThat(chunker.hasNext()).isTrue();
    assertThat(chunker.next()).isNotNull();

    assertThat(chunker.hasNext()).isFalse();

    try {
      chunker.next();
      fail("Should have thrown an exception");
    } catch (NoSuchElementException expected) {
      // Intentionally left empty.
    }
  }

  @Test
  public void emptyData() throws Exception {
    byte[] data = new byte[0];
    Chunker chunker = new Chunker(data);

    assertThat(chunker.hasNext()).isTrue();

    Chunk next = chunker.next();

    assertThat(next).isNotNull();
    assertThat(next.getData()).isEmpty();
    assertThat(next.getOffset()).isEqualTo(0);

    assertThat(chunker.hasNext()).isFalse();

    try {
      chunker.next();
      fail("Should have thrown an exception");
    } catch (NoSuchElementException expected) {
      // Intentionally left empty.
    }
  }

  @Test
  public void reset() throws Exception {
    byte[] data = new byte[]{1, 2, 3};
    Chunker chunker = new Chunker(data, 1);

    assertNextEquals(chunker, (byte) 1);
    assertNextEquals(chunker, (byte) 2);

    chunker.reset();

    assertNextEquals(chunker, (byte) 1);
    assertNextEquals(chunker, (byte) 2);
    assertNextEquals(chunker, (byte) 3);

    chunker.reset();

    assertNextEquals(chunker, (byte) 1);
  }

  private void assertNextEquals(Chunker chunker, byte... data) throws IOException {
    assertThat(chunker.hasNext()).isTrue();
    ByteString next = chunker.next().getData();
    assertThat(next.toByteArray()).isEqualTo(data);
  }
}
