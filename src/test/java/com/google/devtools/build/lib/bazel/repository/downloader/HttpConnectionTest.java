// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.downloader;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.common.net.MediaType;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.nio.charset.Charset;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/**
 * Tests for @{link HttpConnection}.
 */
@RunWith(JUnit4.class)
public class HttpConnectionTest {

  @Test
  public void testEncodingSet() throws Exception {
    Map<String, Charset> charsets = Charset.availableCharsets();
    assertThat(charsets).isNotEmpty();
    Map.Entry<String, Charset> entry = charsets.entrySet().iterator().next();

    String availableEncoding = entry.getKey();
    Charset availableCharset = entry.getValue();

    HttpURLConnection connection = Mockito.mock(HttpURLConnection.class);
    when(connection.getContentEncoding()).thenReturn(availableEncoding);
    Charset charset = HttpConnection.getEncoding(connection);
    assertEquals(availableCharset, charset);
  }

  @Test
  public void testInvalidEncoding() throws Exception {
    HttpURLConnection connection = Mockito.mock(HttpURLConnection.class);
    when(connection.getContentEncoding()).thenReturn("This-isn't-a-valid-content-encoding");
    try {
      HttpConnection.getEncoding(connection);
      fail("Expected exception");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("Got unavailable encoding");
    }
  }

  @Test
  public void testContentType() throws Exception {
    HttpURLConnection connection = Mockito.mock(HttpURLConnection.class);
    when(connection.getContentType()).thenReturn(MediaType.HTML_UTF_8.toString());
    Charset charset = HttpConnection.getEncoding(connection);
    assertEquals(UTF_8, charset);
  }

  @Test
  public void testInvalidContentType() throws Exception {
    HttpURLConnection connection = Mockito.mock(HttpURLConnection.class);
    when(connection.getContentType()).thenReturn("This-isn't-a-valid-content-type");
    try {
      HttpConnection.getEncoding(connection);
      fail("Expected exception");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("Got invalid encoding");
    }
  }

  @Test
  public void testNoEncodingNorContentType() throws Exception {
    HttpURLConnection connection = Mockito.mock(HttpURLConnection.class);
    Charset charset = HttpConnection.getEncoding(connection);
    assertEquals(UTF_8, charset);
  }

  /**
   * Creates a temporary file with the specified {@code fileContents}. The file will be
   * automatically deleted when the JVM exits.
   *
   * @param fileContents the contents of the file
   * @return the {@link File} object representing the temporary file
   */
  private static File createTempFile(byte[] fileContents) throws IOException {
    File temp = File.createTempFile("httpConnectionTest", ".tmp");
    temp.deleteOnExit();
    try (FileOutputStream outputStream = new FileOutputStream(temp)) {
      outputStream.write(fileContents);
    }
    return temp;
  }

  @Test
  public void testLocalFileDownload() throws Exception {
    byte[] fileContents = "this is a test".getBytes(UTF_8);
    File temp = createTempFile(fileContents);
    HttpConnection httpConnection =
        HttpConnection.createAndConnect(temp.toURI().toURL(), ImmutableMap.<String, String>of());

    assertThat(httpConnection.getContentLength()).isEqualTo(fileContents.length);

    byte[] readContents = ByteStreams.toByteArray(httpConnection.getInputStream());
    assertThat(readContents).isEqualTo(fileContents);
  }

  @Test
  public void testLocalEmptyFileDownload() throws Exception {
    byte[] fileContents = new byte[0];
    // create a temp file
    File temp = createTempFile(fileContents);
    try {
      HttpConnection.createAndConnect(temp.toURI().toURL(), ImmutableMap.<String, String>of());
      fail("Expected exception");
    } catch (IOException ex) {
      // expected
    }
  }
}
