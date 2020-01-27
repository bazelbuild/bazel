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

import java.util.Calendar;
import java.util.GregorianCalendar;

@RunWith(JUnit4.class)
public class ZipUtilTest {
  @Rule public ExpectedException thrown = ExpectedException.none();

  @Test public void testShortToLittleEndian() {
    byte[] bytes = ZipUtil.shortToLittleEndian((short) 4660);
    assertThat(bytes).isEqualTo(new byte[]{ 0x34, 0x12 });
  }

  @Test public void testShortToLittleEndian_Signed() {
    byte[] bytes = ZipUtil.shortToLittleEndian((short) -3532);
    assertThat(bytes).isEqualTo(new byte[]{ 0x34, (byte) 0xf2 });
  }

  @Test public void testIntToLittleEndian() {
    byte[] bytes = ZipUtil.intToLittleEndian(305419896);
    assertThat(bytes).isEqualTo(new byte[]{ 0x78, 0x56, 0x34, 0x12 });
  }

  @Test public void testIntToLittleEndian_Signed() {
    byte[] bytes = ZipUtil.intToLittleEndian(-231451016);
    assertThat(bytes).isEqualTo(new byte[]{ 0x78, 0x56, 0x34, (byte) 0xf2 });
  }

  @Test public void testLongToLittleEndian() {
    byte[] bytes = ZipUtil.longToLittleEndian(305419896);
    assertThat(bytes).isEqualTo(new byte[]{ 0x78, 0x56, 0x34, 0x12, 0x0, 0x0, 0x0, 0x0 });
  }

  @Test public void testLongToLittleEndian_Signed() {
    byte[] bytes = ZipUtil.longToLittleEndian(-231451016);
    assertThat(bytes).isEqualTo(new byte[]{ 0x78, 0x56, 0x34, (byte) 0xf2,
        (byte) 0xff, (byte) 0xff, (byte) 0xff, (byte) 0xff });
  }

  @Test public void testGet16() {
    short result = ZipUtil.get16(new byte[]{ 0x34, 0x12 }, 0);
    assertThat(result).isEqualTo((short) 0x1234);
    assertThat(result).isEqualTo((short) 4660);
  }

  @Test public void testGet16_Signed() {
    short result = ZipUtil.get16(new byte[]{ 0x34, (byte) 0xff }, 0);
    assertThat(result).isEqualTo((short) 0xff34);
    assertThat(result).isEqualTo((short) -204);
  }

  @Test public void testGet32() {
    int result = ZipUtil.get32(new byte[]{ 0x78, 0x56, 0x34, 0x12 }, 0);
    assertThat(result).isEqualTo(0x12345678);
    assertThat(result).isEqualTo(305419896);
  }

  @Test public void testGet32_Short() {
    int result = ZipUtil.get32(new byte[]{ 0x34, (byte) 0xff, 0x0, 0x0 }, 0);
    assertThat(result).isEqualTo(0xff34);
    assertThat(result).isEqualTo(65332);
  }

  @Test public void testGet32_Signed() {
    int result = ZipUtil.get32(new byte[]{ 0x34, (byte) 0xff, (byte) 0xff, (byte) 0xff }, 0);
    assertThat(result).isEqualTo(0xffffff34);
    assertThat(result).isEqualTo(-204);
  }

  @Test public void testGetUnsignedShort() {
    int result = ZipUtil.getUnsignedShort(new byte[]{ 0x34, 0x12 }, 0);
    assertThat(result).isEqualTo(0x1234);
    assertThat(result).isEqualTo(4660);
  }

  @Test public void testGetUnsignedShort_Big() {
    int result = ZipUtil.getUnsignedShort(new byte[]{ 0x34, (byte) 0xff }, 0);
    assertThat(result).isEqualTo(0xff34);
    assertThat(result).isEqualTo(65332);
  }

  @Test public void testGetUnsignedInt() {
    long result = ZipUtil.getUnsignedInt(new byte[]{ 0x34, 0x12, 0x0, 0x0 }, 0);
    assertThat(result).isEqualTo(0x1234);
    assertThat(result).isEqualTo(4660);
  }

  @Test public void testGetUnsignedShort_FFFF() {
    int result = ZipUtil.getUnsignedShort(new byte[]{ (byte) 0xff,  (byte) 0xff }, 0);
    assertThat((short) result).isEqualTo((short) -1);
  }

  @Test public void testGetUnsignedInt_Big() {
    long result = ZipUtil.getUnsignedInt(
        new byte[]{ 0x34, (byte) 0xff, (byte) 0xff, (byte) 0xff }, 0);
    assertThat(result).isEqualTo(0xffffff34L);
    assertThat(result).isEqualTo(4294967092L);
  }

  @Test public void testTimeConversion_DosToUnix() {
    int dos = (20 << 25) | (2 << 21) | (14 << 16) | (3 << 11) | (7 << 5) | (15 >> 1);

    Calendar time = new GregorianCalendar(2000, Calendar.FEBRUARY, 14, 3, 7, 14);
    long expectedUnixTime = time.getTimeInMillis();
    assertThat(ZipUtil.dosToUnixTime(dos)).isEqualTo(expectedUnixTime);
  }

  @Test public void testTimeConversion_UnixToDos() {
    Calendar time = new GregorianCalendar(2000, Calendar.FEBRUARY, 14, 3, 7, 14);
    long unix = time.getTimeInMillis();
    int expectedDosTime = (20 << 25) | (2 << 21) | (14 << 16) | (3 << 11) | (7 << 5) | (15 >> 1);
    assertThat(ZipUtil.unixToDosTime(unix)).isEqualTo(expectedDosTime);
  }

  @Test public void testTimeConversion_UnixToDos_LowBound() {
    Calendar time = Calendar.getInstance();
    time.setTimeInMillis(ZipUtil.DOS_EPOCH);
    time.add(Calendar.SECOND, -1);
    thrown.expect(IllegalArgumentException.class);
    ZipUtil.unixToDosTime(time.getTimeInMillis());
  }

  @Test public void testTimeConversion_UnixToDos_HighBound_Rounding() {
    Calendar time = Calendar.getInstance();
    time.setTimeInMillis(ZipUtil.MAX_DOS_DATE);
    ZipUtil.unixToDosTime(time.getTimeInMillis());
  }

  @Test public void testTimeConversion_UnixToDos_HighBound() {
    Calendar time = Calendar.getInstance();
    time.setTimeInMillis(ZipUtil.MAX_DOS_DATE);
    time.add(Calendar.SECOND, 1);
    thrown.expect(IllegalArgumentException.class);
    ZipUtil.unixToDosTime(time.getTimeInMillis());
  }

  @Test public void testTimeConversion_UnixToUnix() {
    Calendar from = new GregorianCalendar(2000, Calendar.FEBRUARY, 14, 3, 7, 15);
    Calendar to = new GregorianCalendar(2000, Calendar.FEBRUARY, 14, 3, 7, 14);
    assertThat(ZipUtil.dosToUnixTime(ZipUtil.unixToDosTime(from.getTimeInMillis())))
        .isEqualTo(to.getTimeInMillis());
  }

  @Test public void testTimeConversion_DosToDos() {
    int dos = (20 << 25) | (2 << 21) | (14 << 16) | (3 << 11) | (7 << 5) | (15 >> 1);
    assertThat(ZipUtil.unixToDosTime(ZipUtil.dosToUnixTime(dos))).isEqualTo(dos);
  }

  @Test public void testTimeConversion_DosToDos_Zero() {
    int dos = 0;
    thrown.expect(IllegalArgumentException.class);
    assertThat(ZipUtil.unixToDosTime(ZipUtil.dosToUnixTime(dos))).isEqualTo(0);
  }
}
