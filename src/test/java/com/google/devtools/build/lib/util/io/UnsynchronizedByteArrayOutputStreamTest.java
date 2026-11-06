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
package com.google.devtools.build.lib.util.io;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link UnsynchronizedByteArrayOutputStream}. */
@RunWith(JUnit4.class)
public final class UnsynchronizedByteArrayOutputStreamTest {

  @Test
  public void emptyStream() {
    var out = new UnsynchronizedByteArrayOutputStream();
    assertThat(out.size()).isEqualTo(0);
    assertThat(out.toByteArray()).isEmpty();
  }

  @Test
  public void negativeInitialSize_throws() {
    assertThrows(IllegalArgumentException.class, () -> new UnsynchronizedByteArrayOutputStream(-1));
  }

  @Test
  public void writeSingleBytes_resizesFromZeroCapacity() {
    var out = new UnsynchronizedByteArrayOutputStream(0);
    out.write(1);
    out.write(2);
    out.write(3);
    assertThat(out.size()).isEqualTo(3);
    assertThat(out.toByteArray()).isEqualTo(new byte[] {1, 2, 3});
  }

  @Test
  public void writeByteArray_resizesAcrossCapacityBoundaries() throws IOException {
    var out = new UnsynchronizedByteArrayOutputStream(2);
    out.write(new byte[] {1, 2});
    out.write(new byte[] {0, 3, 4, 5, 0}, 1, 3);
    assertThat(out.size()).isEqualTo(5);
    assertThat(out.toByteArray()).isEqualTo(new byte[] {1, 2, 3, 4, 5});
  }

  @Test
  public void writeByteArray_invalidBounds_throws() {
    var out = new UnsynchronizedByteArrayOutputStream();
    byte[] data = new byte[] {1, 2, 3};
    assertThrows(NullPointerException.class, () -> out.write(null, 0, 1));
    assertThrows(IndexOutOfBoundsException.class, () -> out.write(data, -1, 1));
    assertThrows(IndexOutOfBoundsException.class, () -> out.write(data, 0, -1));
    assertThrows(IndexOutOfBoundsException.class, () -> out.write(data, 2, 2));
  }

  @Test
  public void reset_discardsContentsAndReusesBuffer() throws IOException {
    var out = new UnsynchronizedByteArrayOutputStream(4);
    out.write(new byte[] {1, 2, 3});
    byte[] first = out.toByteArray();

    out.reset();
    assertThat(out.size()).isEqualTo(0);
    assertThat(out.toByteArray()).isEmpty();

    out.write(new byte[] {4, 5});
    assertThat(out.size()).isEqualTo(2);
    assertThat(out.toByteArray()).isEqualTo(new byte[] {4, 5});
    assertThat(first).isEqualTo(new byte[] {1, 2, 3});
  }
}
