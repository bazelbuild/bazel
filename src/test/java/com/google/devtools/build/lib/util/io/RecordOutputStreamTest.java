// Copyright 2025 The Bazel Authors. All rights reserved.
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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RecordOutputStream}. */
@RunWith(JUnit4.class)
public final class RecordOutputStreamTest {
  @Test
  public void empty() throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (RecordOutputStream recordOut = new RecordOutputStream(baos)) {}
    assertThat(baos.toByteArray()).isEmpty();
  }

  @Test
  public void write_singleRecord() throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (RecordOutputStream recordOut = new RecordOutputStream(baos)) {
      recordOut.write(new byte[] {0x12, 0x34, 0x56});
      recordOut.finishRecord();
    }
    assertThat(baos.toByteArray()).isEqualTo(new byte[] {0x12, 0x34, 0x56});
  }

  @Test
  public void write_multipleRecords() throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (RecordOutputStream recordOut = new RecordOutputStream(baos)) {
      recordOut.write(new byte[] {0x12, 0x34, 0x56});
      recordOut.finishRecord();
      recordOut.write(new byte[] {0x21, 0x43, 0x65});
      recordOut.finishRecord();
    }
    assertThat(baos.toByteArray()).isEqualTo(new byte[] {0x12, 0x34, 0x56, 0x21, 0x43, 0x65});
  }

  @Test
  public void write_largeRecord_singleWrite() throws IOException {
    byte[] record = new byte[65536];
    for (int i = 0; i < record.length; i++) {
      record[i] = (byte) i;
    }
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (RecordOutputStream recordOut = new RecordOutputStream(baos)) {
      recordOut.write(record);
      recordOut.finishRecord();
    }
    assertThat(baos.toByteArray()).isEqualTo(record);
  }

  @Test
  public void write_largeRecord_multipleWrites() throws IOException {
    byte[] record = new byte[65536];
    for (int i = 0; i < record.length; i++) {
      record[i] = (byte) i;
    }
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (RecordOutputStream recordOut = new RecordOutputStream(baos)) {
      for (int i = 0; i < record.length; i++) {
        recordOut.write(record[i]);
      }
      recordOut.finishRecord();
    }
    assertThat(baos.toByteArray()).isEqualTo(record);
  }

  @Test
  public void flush_onlyCompleteRecords() throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (RecordOutputStream recordOut = new RecordOutputStream(baos)) {
      recordOut.write(new byte[] {0x12, 0x34});
      recordOut.finishRecord();
      recordOut.write(new byte[] {0x56, 0x78});
      recordOut.flush();
      assertThat(baos.toByteArray()).isEqualTo(new byte[] {0x12, 0x34});
      recordOut.write(new byte[] {0x21, 0x43});
      recordOut.finishRecord();
      recordOut.flush();
      assertThat(baos.toByteArray()).isEqualTo(new byte[] {0x12, 0x34, 0x56, 0x78, 0x21, 0x43});
    }
  }

  @Test
  public void close_onlyCompleteRecords() throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (RecordOutputStream recordOut = new RecordOutputStream(baos)) {
      recordOut.write(new byte[] {0x21, 0x34});
      recordOut.finishRecord();
      recordOut.write(new byte[] {0x56, 0x78});
    }
    assertThat(baos.toByteArray()).isEqualTo(new byte[] {0x21, 0x34});
  }
}
