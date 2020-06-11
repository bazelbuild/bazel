// Copyright 2020 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link SynchronizedOutputStream}.
 *
 * <p>Note that, at the time of writing, these tests serve to document the actual behavior of the
 * class, not necessarily the desired behavior.
 */
@RunWith(JUnit4.class)
public class SynchronizedOutputStreamTest {

  @Test
  public void testReadAndResetReturnsChunkedWritesSinceLastCall() throws IOException {
    SynchronizedOutputStream underTest =
        new SynchronizedOutputStream(/*maxBufferedLength=*/ 5, /*maxChunkSize=*/ 5);

    underTest.write(new byte[] {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'});
    assertThat(underTest.readAndReset()).containsExactly("abcde", "fgh").inOrder();

    assertThat(underTest.readAndReset()).isEmpty();

    underTest.write(new byte[] {'i', 'j', 'k'});
    assertThat(underTest.readAndReset()).containsExactly("ijk");
  }

  @Test
  public void testWriteFlushesStreamerWhenMaxBufferedLengthReached() throws IOException {
    SynchronizedOutputStream underTest =
        new SynchronizedOutputStream(/*maxBufferedLength=*/ 3, /*maxChunkSize=*/ 3);

    List<List<String>> writes = new ArrayList<>();
    BuildEventStreamer mockStreamer = mock(BuildEventStreamer.class);
    doAnswer(
            inv -> {
              writes.add(ImmutableList.copyOf(underTest.readAndReset()));
              return null;
            })
        .when(mockStreamer)
        .flush();
    underTest.registerStreamer(mockStreamer);

    underTest.write(new byte[] {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'});
    underTest.write(new byte[] {'i', 'j'});
    underTest.write(new byte[] {'k', 'l'});

    assertThat(writes)
        .containsExactly(
            ImmutableList.of("abc", "def", "gh"),
            // The write of {'k', 'j'} would have put the buffer over size, so {'i', 'j'} was
            // flushed.
            ImmutableList.of("ij"))
        .inOrder();
  }

  @Test
  public void testUsesMaxOfMaxBufferedSizeAndMaxChunkSizeForChunking() throws IOException {
    SynchronizedOutputStream underTest =
        new SynchronizedOutputStream(/*maxBufferedLength=*/ 2, /*maxChunkSize=*/ 1);

    underTest.write(new byte[] {'a', 'b', 'c', 'd'});
    assertThat(underTest.readAndReset()).containsExactly("ab", "cd").inOrder();
  }

  // TODO(b/154242266): Make BES handle binary data correctly.
  @Ignore("b/154242266 - BES doesn't handle binary stdout/err correctly")
  @Test
  public void testHandlesArbitraryBinaryDataCorrectly() throws IOException {
    SynchronizedOutputStream underTest =
        new SynchronizedOutputStream(/*maxBufferedLength=*/ 1, /*maxChunkSize=*/ 1);

    byte[] input = new byte[] {(byte) 0xff};
    underTest.write(input);

    String result = Iterables.getOnlyElement(underTest.readAndReset());
    // In the real code the result eventually winds up in a protobuf, which treats strings as utf8.
    assertThat(result.getBytes(StandardCharsets.UTF_8)).isEqualTo(input);
  }
}
