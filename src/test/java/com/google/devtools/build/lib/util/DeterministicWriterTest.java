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

package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.CountDownLatch;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the bounded streams produced by {@link DeterministicWriter}. */
@RunWith(JUnit4.class)
public final class DeterministicWriterTest {
  @Test
  public void readsAreRepeatableAcrossBufferBoundaries() throws Exception {
    byte[] contents = "héllo 🌍".repeat(100).getBytes(UTF_8);
    DeterministicWriter writer = out -> out.write(contents);
    for (int bufferSize : new int[] {1, 17, 1024}) {
      try (InputStream in = writer.getInputStream(bufferSize)) {
        assertThat(in.read()).isEqualTo(contents[0]);
        assertThat(in.readAllBytes())
            .isEqualTo(java.util.Arrays.copyOfRange(contents, 1, contents.length));
        assertThat(in.read()).isEqualTo(-1);
      }
    }
  }

  @Test
  public void emptyWriterReturnsEof() throws Exception {
    DeterministicWriter writer = out -> {};
    try (InputStream in = writer.getInputStream(1)) {
      assertThat(in.read()).isEqualTo(-1);
    }
  }

  @Test
  public void writerFailureIsNotReportedAsEof() throws Exception {
    IOException failure = new IOException("writer failed");
    DeterministicWriter writer =
        out -> {
          out.write(42);
          throw failure;
        };
    try (InputStream in = writer.getInputStream(1)) {
      assertThat(in.read()).isEqualTo(42);
      assertThat(assertThrows(IOException.class, in::read))
          .hasCauseThat()
          .isSameInstanceAs(failure);
    }
  }

  @Test
  public void uncheckedWriterFailureIsPropagatedToBulkRead() throws Exception {
    RuntimeException failure = new IllegalStateException("writer failed");
    DeterministicWriter writer =
        out -> {
          throw failure;
        };
    try (InputStream in = writer.getInputStream(1)) {
      assertThat(assertThrows(IOException.class, in::readAllBytes))
          .hasCauseThat()
          .isSameInstanceAs(failure);
    }
  }

  @Test
  public void closeStopsWriterBlockedOnFullBuffer() throws Exception {
    CountDownLatch bufferFilled = new CountDownLatch(1);
    CountDownLatch writerStopped = new CountDownLatch(1);
    DeterministicWriter writer =
        out -> {
          try {
            out.write(1);
            bufferFilled.countDown();
            out.write(2);
          } finally {
            writerStopped.countDown();
          }
        };
    try (InputStream in = writer.getInputStream(1)) {
      assertThat(bufferFilled.await(10, SECONDS)).isTrue();
      assertThat(writerStopped.getCount()).isEqualTo(1);
    }
    assertThat(writerStopped.await(10, SECONDS)).isTrue();
  }
}
