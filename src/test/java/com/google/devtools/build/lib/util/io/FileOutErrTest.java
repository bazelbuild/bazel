// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.util.io.FileOutErr.FileRecordingOutputStream;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link OutErr}. */
@RunWith(JUnit4.class)
public class FileOutErrTest {

  private FileSystem fs;

  @Before
  public void setUp() {
    fs = new InMemoryFileSystem();
  }

  private FileRecordingOutputStream newFileRecordingOutputStream(String path) {
    Path outputFile = fs.getPath(path);
    return new FileRecordingOutputStream(outputFile);
  }

  @Test
  public void testFileRecordingOutputStream_doesNotExistByDefault() throws Exception {
    FileRecordingOutputStream os = newFileRecordingOutputStream("/some-file.txt");

    assertThat(os.hasRecordedOutput()).isFalse();
    assertThat(os.getRecordedOutput()).isEmpty();
    assertThat(os.getRecordedOutputSize()).isEqualTo(0);

    ByteArrayOutputStream recorder = new ByteArrayOutputStream();
    os.dumpOut(recorder);
    assertThat(recorder.toByteArray()).isEmpty();

    // Existence and error checks must come last to ensure previous calls have no side-effects.
    assertThat(os.getFileUnsafe().exists()).isFalse();
    assertThat(os.hadError()).isFalse();
  }

  @Test
  public void testFileRecordingOutputStream_createOutOfBandAsEmpty() throws Exception {
    FileRecordingOutputStream os = newFileRecordingOutputStream("/some-file.txt");
    Path path = os.getFile();
    path.getOutputStream().close();

    assertThat(os.hasRecordedOutput()).isFalse();
    assertThat(os.getRecordedOutput()).isEmpty();
    assertThat(os.getRecordedOutputSize()).isEqualTo(0);

    ByteArrayOutputStream recorder = new ByteArrayOutputStream();
    os.dumpOut(recorder);
    assertThat(recorder.toByteArray()).isEmpty();

    // Existence and error checks must come last to ensure previous calls have no side-effects.
    assertThat(os.getFileUnsafe().exists()).isTrue();
    assertThat(os.hadError()).isFalse();
  }

  @Test
  public void testFileRecordingOutputStream_createOutOfBandWithContents() throws Exception {
    FileRecordingOutputStream os = newFileRecordingOutputStream("/some-file.txt");
    Path path = os.getFile();
    byte[] data = "12345".getBytes(StandardCharsets.ISO_8859_1);
    try (OutputStream writer = path.getOutputStream()) {
      writer.write(data);
    }

    assertThat(os.hasRecordedOutput()).isTrue();
    assertThat(os.getRecordedOutput()).isEqualTo(data);
    assertThat(os.getRecordedOutputSize()).isEqualTo(data.length);

    ByteArrayOutputStream recorder = new ByteArrayOutputStream();
    os.dumpOut(recorder);
    assertThat(recorder.toByteArray()).isEqualTo(data);

    // Existence and error checks must come last to ensure previous calls have no side-effects.
    assertThat(os.getFileUnsafe().exists()).isTrue();
    assertThat(os.hadError()).isFalse();
  }

  @Test
  public void testFileRecordingOutputStream_write() throws Exception {
    FileRecordingOutputStream os = newFileRecordingOutputStream("/some-file.txt");
    byte[] data = "12345".getBytes(StandardCharsets.ISO_8859_1);
    try {
      os.write(data);
    } finally {
      os.close();
    }

    assertThat(os.hasRecordedOutput()).isTrue();
    assertThat(os.getRecordedOutput()).isEqualTo(data);
    assertThat(os.getRecordedOutputSize()).isEqualTo(data.length);

    ByteArrayOutputStream recorder = new ByteArrayOutputStream();
    os.dumpOut(recorder);
    assertThat(recorder.toByteArray()).isEqualTo(data);

    // Existence and error checks must come last to ensure previous calls have no side-effects.
    assertThat(os.getFileUnsafe().exists()).isTrue();
    assertThat(os.hadError()).isFalse();
  }

  @Test
  public void testFileRecordingOutputStream_clearAfterCreation() throws Exception {
    FileRecordingOutputStream os = newFileRecordingOutputStream("/some-file.txt");
    Path path = os.getFile();
    try {
      os.write("12345".getBytes(StandardCharsets.ISO_8859_1));
    } finally {
      os.close();
    }

    assertThat(path.exists()).isTrue();
    assertThat(os.getRecordedOutputSize()).isGreaterThan(0);
    assertThat(os.hadError()).isFalse();
    os.clear();
    assertThat(path.exists()).isFalse();
    assertThat(os.getRecordedOutputSize()).isEqualTo(0);
    assertThat(os.hadError()).isFalse();
  }

  @Test
  public void testFileRecordingOutputStream_errorDuringSizeCheck() throws Exception {
    fs.getPath("/dir").createDirectory();
    FileRecordingOutputStream os = newFileRecordingOutputStream("/dir/some-file.txt");
    Path path = os.getFile();
    path.getOutputStream().close();
    fs.getPath("/dir").setReadable(false);
    fs.getPath("/dir").setExecutable(false);

    IOException expected = assertThrows(IOException.class, os::getRecordedOutputSize);
    assertThat(expected.toString()).contains("Permission denied");

    ByteArrayOutputStream recorder = new ByteArrayOutputStream();
    os.dumpOut(recorder);
    assertThat(new String(recorder.toByteArray(), StandardCharsets.ISO_8859_1))
        .contains("Permission denied");

    // Restore directory permissions so our existence check works.
    fs.getPath("/dir").setReadable(true);
    fs.getPath("/dir").setExecutable(true);
    // Existence and error checks must come last to ensure previous calls have no side-effects.
    assertThat(os.getFileUnsafe().exists()).isTrue();
    assertThat(os.hadError()).isTrue();
  }

  @Test
  public void testFileRecordingOutputStream_errorDuringRead() throws Exception {
    FileRecordingOutputStream os = newFileRecordingOutputStream("/some-file.txt");
    Path path = os.getFile();
    path.getOutputStream().close();
    path.setReadable(false);

    String error = new String(os.getRecordedOutput(), StandardCharsets.ISO_8859_1);
    // The error message comes from the system so we cannot be too specific about what we look for.
    assertThat(error).contains("Permission denied");
    assertThat(os.getRecordedOutputSize()).isGreaterThan(0);

    ByteArrayOutputStream recorder = new ByteArrayOutputStream();
    os.dumpOut(recorder);
    assertThat(recorder.toByteArray())
        .isEqualTo((error + "\n" + error).getBytes(StandardCharsets.ISO_8859_1));

    // Existence and error checks must come last to ensure previous calls have no side-effects.
    assertThat(os.getFileUnsafe().exists()).isTrue();
    assertThat(os.hadError()).isTrue();
  }
}
