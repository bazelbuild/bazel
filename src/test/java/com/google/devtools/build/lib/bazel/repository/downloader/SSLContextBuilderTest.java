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

package com.google.devtools.build.lib.bazel.repository.downloader;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link SSLContextBuilder}.
 */
@RunWith(JUnit4.class)
public class SSLContextBuilderTest {

  @Test
  public void buildWithoutCertificateNorKeyReturnsNull() throws Exception {
    AuthAndTLSOptions options = new AuthAndTLSOptions();
    assertThat(SSLContextBuilder.build(options)).isNull();
  }

  @Test
  public void buildWithoutCertificateReturnsNull() throws Exception {
    AuthAndTLSOptions options = new AuthAndTLSOptions();
    options.tlsClientKey = "some value";
    assertThat(SSLContextBuilder.build(options)).isNull();
  }

  @Test
  public void buildWithoutKeyReturnsNull() throws Exception {
    AuthAndTLSOptions options = new AuthAndTLSOptions();
    options.tlsClientCertificate = "some value";
    assertThat(SSLContextBuilder.build(options)).isNull();
  }

  private void doParseChunksValidTest(String lines, String... expectedChunks) throws Exception {
    File tempFile = File.createTempFile("SSLContextBuilderTest", "");
    tempFile.deleteOnExit();

    try (final PrintWriter out = new PrintWriter(new FileWriter(tempFile))) {
      out.print(lines);
    }

    List<byte[]> byteChunks = SSLContextBuilder.parseChunks(tempFile.getPath(), "CHUNK");
    List<String> stringChunks = new ArrayList<>();
    for (byte[] byteChunk : byteChunks) {
      stringChunks.add(new String(byteChunk, StandardCharsets.UTF_8));
    }

    assertThat(stringChunks).containsExactly((Object[]) expectedChunks);
  }

  @Test
  public void parseChunksOneValid() throws Exception {
    doParseChunksValidTest("-----BEGIN CHUNK-----\nSGVsbG8sIFRMUyEK\n-----END CHUNK-----\n",
        "Hello, TLS!\n");
  }

  @Test
  public void parseChunksTwoValid() throws Exception {
    doParseChunksValidTest(
        "-----BEGIN CHUNK-----\nSGVsbG8sIFRMUyEK\n-----END CHUNK-----\n-----BEGIN CHUNK-----\nQnllLCBUTFMhCg==\n-----END CHUNK-----\n",
        "Hello, TLS!\n", "Bye, TLS!\n");
  }

  private void doParseChunksMalformedTest(String lines,
      Class<? extends Throwable> expectedThrowable, String expectedMessage) throws Exception {
    File tempFile = File.createTempFile("SSLContextBuilderTest", "");
    tempFile.deleteOnExit();

    try (final PrintWriter out = new PrintWriter(new FileWriter(tempFile))) {
      out.print(lines);
    }

    assertThat(
        assertThrows(expectedThrowable,
            () -> SSLContextBuilder.parseChunks(tempFile.getPath(), "CHUNK")))
        .hasMessageThat().contains(expectedMessage);
  }

  @Test
  public void parseChunksEmptyFileIsError() throws Exception {
    doParseChunksMalformedTest("", IOException.class, "content without BEGIN");
  }

  @Test
  public void parseChunksBeginWithoutEndIsError() throws Exception {
    doParseChunksMalformedTest("-----BEGIN CHUNK-----\n", IOException.class,
        "BEGIN tag without END");
  }

  @Test
  public void parseChunksContentWithoutBeginIsError() throws Exception {
    doParseChunksMalformedTest("12345\n", IOException.class, "content without BEGIN");
  }

  @Test
  public void parseChunksEndWithoutBeginIsError() throws Exception {
    doParseChunksMalformedTest("-----END CHUNK-----\n", IOException.class, "END tag without BEGIN");
  }

  @Test
  public void parseChunksNestedBeginIsError() throws Exception {
    doParseChunksMalformedTest(
        "-----BEGIN CHUNK-----\n123\n-----BEGIN CHUNK-----\n12345\n-----END CHUNK-----\n",
        IOException.class, "nested BEGIN tags");
  }

  @Test
  public void parseChunksBadBase64IsError() throws Exception {
    doParseChunksMalformedTest(
        "-----BEGIN CHUNK-----\nInvalid Content\n-----END CHUNK-----\n",
        IllegalArgumentException.class, "Illegal base64 character");
  }

  @Test
  public void parseCertificates() throws Exception {
    Certificate[] certs = SSLContextBuilder.parseCertificates(
        "src/test/java/com/google/devtools/build/lib/bazel/repository/downloader/test-tls.crt");
    assertThat(certs.length).isEqualTo(1);
    Certificate cert = certs[0];
    assertThat(cert.getType()).isEqualTo("X.509");
  }

  @Test
  public void parsePrivateKey() throws Exception {
    PrivateKey key = SSLContextBuilder.parsePrivateKey(
        "src/test/java/com/google/devtools/build/lib/bazel/repository/downloader/test-tls.key");
    assertThat(key.getFormat()).isEqualTo("PKCS#8");
  }
}
