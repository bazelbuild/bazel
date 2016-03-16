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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.when;

import com.google.common.net.MediaType;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Map;

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
    assertEquals(StandardCharsets.UTF_8, charset);
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
    assertEquals(StandardCharsets.UTF_8, charset);
  }
}
