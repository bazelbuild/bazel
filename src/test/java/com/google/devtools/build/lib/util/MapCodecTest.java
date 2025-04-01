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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link MapCodec}. */
@RunWith(JUnit4.class)
public final class MapCodecTest {

  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final Path testPath = fs.getPath("/test");

  private static final MapCodec<Integer, Integer> TEST_CODEC =
      new MapCodec<Integer, Integer>() {
        @Override
        protected Integer readKey(DataInput in) throws IOException {
          return in.readInt();
        }

        @Override
        protected Integer readValue(DataInput in) throws IOException {
          return in.readInt();
        }

        @Override
        protected void writeKey(Integer key, DataOutput out) throws IOException {
          out.writeInt(key);
        }

        @Override
        protected void writeValue(Integer value, DataOutput out) throws IOException {
          out.writeInt(value);
        }
      };

  @Test
  public void createWriter_overwriteMissingFileWithEmpty() throws IOException {
    try (var out = TEST_CODEC.createWriter(testPath, 0x42, /* overwrite= */ true)) {}

    assertByteContents(testPath, "00000000 20071105 00000000 00000042");
  }

  @Test
  public void createWriter_overwriteMissingFileWithNonEmpty() throws IOException {
    try (var out = TEST_CODEC.createWriter(testPath, 0x42, /* overwrite= */ true)) {
      out.writeEntry(0x12345678, 0x87654321);
      out.writeEntry(0xabcdef, null);
    }

    assertByteContents(
        testPath, "00000000 20071105 00000000 00000042 fe 12345678 01 87654321 fe 00abcdef 00");
  }

  @Test
  public void createWriter_overwriteExistingFile() throws IOException {
    writeByteContents(testPath, "00000000 20071105 00000000 00000042 fe 12345678 01 87654321");

    try (var out = TEST_CODEC.createWriter(testPath, 0x42, /* overwrite= */ true)) {
      out.writeEntry(0x87654321, 0x12345678);
    }

    assertByteContents(testPath, "00000000 20071105 00000000 00000042 fe 87654321 01 12345678");
  }

  @Test
  public void writer_appendToMissingFile() throws IOException {
    try (var out = TEST_CODEC.createWriter(testPath, 0x42, /* overwrite= */ false)) {
      out.writeEntry(0x12345678, 0x87654321);
    }

    assertByteContents(testPath, "00000000 20071105 00000000 00000042 fe 12345678 01 87654321");
  }

  @Test
  public void writer_appendToExistingEmptyFile() throws IOException {
    writeByteContents(testPath, "00000000 20071105 00000000 00000042");

    try (var out = TEST_CODEC.createWriter(testPath, 0x42, /* overwrite= */ false)) {
      out.writeEntry(0x87654321, 0x12345678);
    }

    assertByteContents(testPath, "00000000 20071105 00000000 00000042 fe 87654321 01 12345678");
  }

  @Test
  public void writer_appendToExistingNonEmptyFile() throws IOException {
    writeByteContents(testPath, "00000000 20071105 00000000 00000042 fe 12345678 01 87654321");

    try (var out = TEST_CODEC.createWriter(testPath, 0x42, /* overwrite= */ false)) {
      out.writeEntry(0x87654321, 0x12345678);
    }

    assertByteContents(
        testPath,
        "00000000 20071105 00000000 00000042 fe 12345678 01 87654321 fe 87654321 01 12345678");
  }

  @Test
  public void createReader_emptyFile() throws IOException {
    writeByteContents(testPath, "00000000 20071105 00000000 00000042");

    try (var in = TEST_CODEC.createReader(testPath, 0x42)) {
      assertThat(in.readEntry()).isNull();
    }
  }

  @Test
  public void createReader_nonEmptyFile() throws IOException {
    writeByteContents(
        testPath, "00000000 20071105 00000000 00000042 fe 12345678 01 87654321 fe 00abcdef 00");

    try (var in = TEST_CODEC.createReader(testPath, 0x42)) {
      assertThat(in.readEntry()).isEqualTo(new MapCodec.Entry<>(0x12345678, 0x87654321));
      assertThat(in.readEntry()).isEqualTo(new MapCodec.Entry<>(0xabcdef, null));
      assertThat(in.readEntry()).isNull();
    }
  }

  @Test
  public void createReader_missingFile() throws IOException {
    IOException e = assertThrows(IOException.class, () -> TEST_CODEC.createReader(testPath, 0x42));
    assertThat(e).hasMessageThat().contains("No such file or directory");
  }

  @Test
  public void createReader_badMagic() throws IOException {
    writeByteContents(testPath, "00000000 12345678 00000000 00000042");

    IOException e = assertThrows(IOException.class, () -> TEST_CODEC.createReader(testPath, 0x42));
    assertThat(e).hasMessageThat().contains("Bad magic number");
  }

  @Test
  public void createReader_incompatibleVersion() throws IOException {
    writeByteContents(testPath, "00000000 20071105 00000000 00000042");

    IOException e = assertThrows(IOException.class, () -> TEST_CODEC.createReader(testPath, 0x43));
    assertThat(e).hasMessageThat().contains("Incompatible version");
  }

  @Test
  public void createReader_corruptedEntry() throws IOException {
    writeByteContents(
        testPath,
        "00000000 20071105 00000000 00000042 fe 12345678 01 87654321 ff 11111111 01 22222222");

    try (var in = TEST_CODEC.createReader(testPath, 0x42)) {
      assertThat(in.readEntry()).isEqualTo(new MapCodec.Entry<>(0x12345678, 0x87654321));
      IOException e = assertThrows(IOException.class, in::readEntry);
      assertThat(e).hasMessageThat().contains("Corrupted entry");
    }
  }

  private static void writeByteContents(Path path, String hex) throws IOException {
    byte[] content = BaseEncoding.base16().lowerCase().decode(hex.replace(" ", ""));
    FileSystemUtils.writeContent(path, content);
  }

  private static void assertByteContents(Path path, String hex) throws IOException {
    String actual = BaseEncoding.base16().lowerCase().encode(FileSystemUtils.readContent(path));
    String expected = hex.replace(" ", "");
    assertThat(actual).isEqualTo(expected);
  }
}
