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

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ExtraDataTest {
  @Rule public ExpectedException thrown = ExpectedException.none();

  @Test public void testFromData() {
    short id = (short) 0xcafe;
    byte[] data = new byte[] { (byte) 0xaa, (byte) 0xbb, (byte) 0xcc };
    // Contains 1 record: 0xcafe 0x0003 0xaa 0xbb 0xcc in little endian
    byte[] record = new byte[] { (byte) 0xfe, (byte) 0xca, 0x03, 0x00, (byte) 0xaa, (byte) 0xbb,
        (byte) 0xcc };
    ExtraData extra = new ExtraData(id, data);
    assertThat(extra.getId()).isEqualTo(id);
    assertThat(extra.getLength()).isEqualTo(record.length);
    assertThat(extra.getData()).isEqualTo(data);
    assertThat(extra.getDataLength()).isEqualTo(data.length);
    assertThat(extra.getBytes()).isEqualTo(record);
    assertThat(extra.getByte(2)).isEqualTo(record[2]);
  }

  @Test public void testFromArray() {
    // Contains 1 record: 0xcafe 0x0004 deadbeef in little endian with 4 bytes padding on the front
    // and 4 bytes padding on the end
    byte[] buf = new byte[] { 0x00, 0x11, 0x22, 0x33, (byte) 0xfe, (byte) 0xca, 0x04, 0x00,
        (byte) 0xde, (byte) 0xad, (byte) 0xbe, (byte) 0xef, (byte) 0xcc, (byte) 0xdd, (byte) 0xee,
        (byte) 0xff };
    // record id: cafe
    short id = (short) 0xcafe;
    // record payload: deadbeef
    byte[] data = new byte[] { (byte) 0xde, (byte) 0xad, (byte) 0xbe, (byte) 0xef };
    // complete record: 0xcafe 0x0004 deadbeef
    byte[] record = new byte[] { (byte) 0xfe, (byte) 0xca, 0x04, 0x00, (byte) 0xde, (byte) 0xad,
        (byte) 0xbe, (byte) 0xef };
    ExtraData extra = new ExtraData(buf, 4);
    assertThat(extra.getId()).isEqualTo(id);
    assertThat(extra.getLength()).isEqualTo(record.length);
    assertThat(extra.getData()).isEqualTo(data);
    assertThat(extra.getDataLength()).isEqualTo(data.length);
    assertThat(extra.getBytes()).isEqualTo(record);
    assertThat(extra.getByte(2)).isEqualTo(record[2]);
  }

  @Test public void testFromArray_shortHeader() {
    thrown.expect(IllegalArgumentException.class);
    thrown.expectMessage("incomplete extra data entry in buffer");
    new ExtraData(new byte[] { (byte) 0xfe,  (byte) 0xca, 0x01 }, 0);
  }

  @Test public void testFromArray_shortData() {
    thrown.expect(IllegalArgumentException.class);
    thrown.expectMessage("incomplete extra data entry in buffer");
    new ExtraData(new byte[] { (byte) 0xfe,  (byte) 0xca, 0x03, 0x00, 0x00 }, 0);
  }
}
