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

package com.google.devtools.build.zip;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.testing.NullPointerTester;
import java.io.IOException;
import java.io.InputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ExtraDataListTest {

  @Test
  public void testNulls() {
    NullPointerTester tester = new NullPointerTester();
    tester.testAllPublicConstructors(ExtraDataList.class);
    tester.testAllPublicInstanceMethods(new ExtraDataList());
  }

  @Test public void testConstructFromList() {
    ExtraData[] extras = new ExtraData[] {
        new ExtraData((short) 0xcafe, new byte[] { 0x00, 0x11, 0x22 }),
        new ExtraData((short) 0xbeef, new byte[] { 0x33, 0x44, 0x55 })
    };

    ExtraDataList extra = new ExtraDataList(extras);
    // Expect 0xcafe 0x0003 0x00 0x11 0x22, 0xbeef 0x0003 0x33 0x44 0x55 in little endian
    assertThat(extra.getBytes()).isEqualTo(new byte[] { (byte) 0xfe, (byte) 0xca, 0x03, 0x00,
        0x00, 0x11, 0x22, (byte) 0xef, (byte) 0xbe, 0x03, 0x00, 0x33, 0x44, 0x55 });
    assertThat(extra.contains((short) 0xcafe)).isTrue();
    assertThat(extra.contains((short) 0xbeef)).isTrue();

    ExtraData cafe = extra.remove((short) 0xcafe);
    // Expect 0xbeef 0x0003 0x33 0x44 0x55 in little endian
    assertThat(extra.getBytes()).isEqualTo(new byte[] { (byte) 0xef, (byte) 0xbe, 0x03, 0x00, 0x33,
        0x44, 0x55 });
    assertThat(extra.contains((short) 0xcafe)).isFalse();
    assertThat(extra.contains((short) 0xbeef)).isTrue();

    extra.add(cafe);
    // Expect 0xbeef 0x0003 0x33 0x44 0x55, 0xcafe 0x0003 0x00 0x11 0x22 in little endian
    assertThat(extra.getBytes()).isEqualTo(new byte[] { (byte) 0xef, (byte) 0xbe, 0x03, 0x00, 0x33,
        0x44, 0x55, (byte) 0xfe, (byte) 0xca, 0x03, 0x00, 0x00, 0x11, 0x22 });
    assertThat(extra.contains((short) 0xcafe)).isTrue();
    assertThat(extra.contains((short) 0xbeef)).isTrue();

    ExtraData beef = extra.get((short) 0xbeef);
    assertThat(beef.getId()).isEqualTo((short) 0xbeef);
  }

  @Test public void testConstructFromBuffer() {
    byte[] buffer = new byte[] { (byte) 0xfe, (byte) 0xca, 0x03, 0x00, 0x00, 0x11, 0x22,
        (byte) 0xef, (byte) 0xbe, 0x03, 0x00, 0x33, 0x44, 0x55 };

    ExtraDataList extra = new ExtraDataList(buffer);
    // Expect 0xcafe 0x0003 0x00 0x11 0x22, 0xbeef 0x0003 0x33 0x44 0x55 in little endian
    assertThat(extra.getBytes()).isEqualTo(new byte[] { (byte) 0xfe, (byte) 0xca, 0x03, 0x00,
        0x00, 0x11, 0x22, (byte) 0xef, (byte) 0xbe, 0x03, 0x00, 0x33, 0x44, 0x55 });
    assertThat(extra.contains((short) 0xcafe)).isTrue();
    assertThat(extra.contains((short) 0xbeef)).isTrue();

    ExtraData cafe = extra.remove((short) 0xcafe);
    // Expect 0xbeef 0x0003 0x33 0x44 0x55 in little endian
    assertThat(extra.getBytes()).isEqualTo(new byte[] { (byte) 0xef, (byte) 0xbe, 0x03, 0x00, 0x33,
        0x44, 0x55 });
    assertThat(extra.contains((short) 0xcafe)).isFalse();
    assertThat(extra.contains((short) 0xbeef)).isTrue();

    extra.add(cafe);
    // Expect 0xbeef 0x0003 0x33 0x44 0x55, 0xcafe 0x0003 0x00 0x11 0x22 in little endian
    assertThat(extra.getBytes()).isEqualTo(new byte[] { (byte) 0xef, (byte) 0xbe, 0x03, 0x00, 0x33,
        0x44, 0x55, (byte) 0xfe, (byte) 0xca, 0x03, 0x00, 0x00, 0x11, 0x22 });
    assertThat(extra.contains((short) 0xcafe)).isTrue();
    assertThat(extra.contains((short) 0xbeef)).isTrue();

    ExtraData beef = extra.get((short) 0xbeef);
    assertThat(beef.getId()).isEqualTo((short) 0xbeef);
  }

  @Test public void testByteStream() throws IOException {
    byte[] buffer = new byte[] { (byte) 0xfe, (byte) 0xca, 0x03, 0x00, 0x00, 0x11, 0x22,
        (byte) 0xef, (byte) 0xbe, 0x03, 0x00, 0x33, 0x44, 0x55 };

    ExtraDataList extra = new ExtraDataList(buffer);
    byte[] bytes = new byte[7];
    InputStream in = extra.getByteStream();
    in.read(bytes);
    // Expect 0xcafe 0x0003 0x00 0x11 0x22 in little endian
    assertThat(bytes).isEqualTo(new byte[] { (byte) 0xfe, (byte) 0xca, 0x03, 0x00, 0x00, 0x11,
        0x22 });
    in.read(bytes);
    // Expect 0xbeef 0x0003 0x33 0x44 0x55 in little endian
    assertThat(bytes).isEqualTo(new byte[] { (byte) 0xef, (byte) 0xbe, 0x03, 0x00, 0x33, 0x44,
        0x55 });
    assertThat(in.read(bytes)).isEqualTo(-1);
  }
}
