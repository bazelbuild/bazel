// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.worker;

import static com.google.common.truth.Truth.assertThat;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the RecordingInputStream class. */
@RunWith(JUnit4.class)
public class RecordingInputStreamTest {

  @Test
  public void getRecordedDataAsString_returnsPlainStringsAsStrings() throws IOException {
    String s = "A good string\nWith two lines\n";
    ByteArrayInputStream bais = new ByteArrayInputStream(s.getBytes(StandardCharsets.UTF_8));
    RecordingInputStream in = new RecordingInputStream(bais);

    in.startRecording(1000);
    in.readRemaining();

    assertThat(in.getRecordedDataAsString()).isEqualTo(s);
  }

  @Test
  public void getRecordedDataAsString_returnsNonUtf8AsHex() throws IOException {
    ByteArrayInputStream bais =
        new ByteArrayInputStream(new byte[] {(byte) 0xFF, (byte) 0xFE, 0X01});
    RecordingInputStream in = new RecordingInputStream(bais);
    byte[] inBuf = new byte[1000];

    in.startRecording(1000);
    in.read(inBuf);

    assertThat(in.getRecordedDataAsString())
        .isEqualTo(
            "Not UTF-8, printing as hex\n"
                + "FF FE 01                                          |...              |\n");
  }

  @Test
  public void getRecordedDataAsString_returnsMixedAsHex() throws IOException {
    String s = "One 17-char line!";
    // Doubles the length on each iteration
    for (int i = 0; i < 6; i++) {
      s += s;
    }
    byte[] bytes = s.getBytes(StandardCharsets.US_ASCII);
    bytes[0] = 0x00;
    bytes[1] = 0x01;
    ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
    RecordingInputStream in = new RecordingInputStream(bais);
    byte[] inBuf = new byte[1025];

    in.startRecording(1025);
    in.read(inBuf);

    assertThat(in.getRecordedDataAsString())
        .startsWith(
            "Not UTF-8, printing first 1024 bytes as hex\n"
                + "00 01 65 20 31 37 2D 63  68 61 72 20 6C 69 6E 65  |..e 17-c har line|");
  }

  @Test
  public void getRecordedDataAsString_returnsLongNonUtf8AsHexLines() throws IOException {
    byte[] buf = new byte[25];
    for (int i = 0; i < 25; i++) {
      buf[i] = (byte) i;
    }
    buf[0] = (byte) 0xFF;
    buf[1] = (byte) 0xFE;
    ByteArrayInputStream bais = new ByteArrayInputStream(buf);
    RecordingInputStream in = new RecordingInputStream(bais);
    byte[] inBuf = new byte[1000];

    in.startRecording(1000);
    in.read(inBuf);

    assertThat(in.getRecordedDataAsString())
        .startsWith(
            "Not UTF-8, printing as hex\n"
                + "FF FE 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F  |........ ........|\n"
                + "10 11 12 13 14 15 16 17  18                       |........ .       |\n");
  }
}
