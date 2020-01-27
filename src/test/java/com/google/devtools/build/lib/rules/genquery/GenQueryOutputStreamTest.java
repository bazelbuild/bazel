// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.genquery;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Strings;
import com.google.devtools.build.lib.rules.genquery.GenQueryOutputStream.GenQueryResult;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.protobuf.ByteString;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link GenQueryOutputStream}. */
@RunWith(JUnit4.class)
public class GenQueryOutputStreamTest {

  @Test
  public void testSmallOutputMultibyteWriteWithCompressionEnabled() throws IOException {
    runMultibyteWriteTest(
        Strings.repeat("xyz", 10_000),
        /*compressionEnabled=*/ true,
        GenQueryOutputStream.RegularResult.class);
  }

  @Test
  public void testSmallOutputMultibyteWriteWithCompressionDisabled() throws IOException {
    runMultibyteWriteTest(
        Strings.repeat("xyz", 10_000),
        /*compressionEnabled=*/ false,
        GenQueryOutputStream.RegularResult.class);
  }

  @Test
  public void testBigOutputMultibyteWriteWithCompressionEnabled() throws IOException {
    runMultibyteWriteTest(
        Strings.repeat("xyz", 1_000_000),
        /*compressionEnabled=*/ true,
        GenQueryOutputStream.CompressedResult.class);
  }

  @Test
  public void testBigOutputMultibyteWriteWithCompressionDisabled() throws IOException {
    runMultibyteWriteTest(
        Strings.repeat("xyz", 1_000_000),
        /*compressionEnabled=*/ false,
        GenQueryOutputStream.RegularResult.class);
  }

  @Test
  public void testSmallOutputSingleByteWritesWithCompressionEnabled() throws IOException {
    runSingleByteWriteTest(
        Strings.repeat("xyz", 10_000),
        /*compressionEnabled=*/ true,
        GenQueryOutputStream.RegularResult.class);
  }

  @Test
  public void testSmallOutputSingleByteWritesWithCompressionDisabled() throws IOException {
    runSingleByteWriteTest(
        Strings.repeat("xyz", 10_000),
        /*compressionEnabled=*/ false,
        GenQueryOutputStream.RegularResult.class);
  }

  @Test
  public void testBigOutputSingleByteWritesWithCompressionEnabled() throws IOException {
    runSingleByteWriteTest(
        Strings.repeat("xyz", 1_000_000),
        /*compressionEnabled=*/ true,
        GenQueryOutputStream.CompressedResult.class);
  }

  @Test
  public void testBigOutputSingleByteWritesWithCompressionDisabled() throws IOException {
    runSingleByteWriteTest(
        Strings.repeat("xyz", 1_000_000),
        /*compressionEnabled=*/ false,
        GenQueryOutputStream.RegularResult.class);
  }

  private static void runMultibyteWriteTest(
      String data, boolean compressionEnabled, Class<? extends GenQueryResult> resultClass)
      throws IOException {
    GenQueryOutputStream underTest = new GenQueryOutputStream(compressionEnabled);
    underTest.write(data.getBytes(StandardCharsets.UTF_8));
    underTest.close();

    GenQueryOutputStream.GenQueryResult result = underTest.getResult();
    assertThat(result).isInstanceOf(resultClass);
    assertThat(result.getBytes()).isEqualTo(ByteString.copyFromUtf8(data));
    assertThat(result.size()).isEqualTo(data.length());

    Fingerprint fingerprint = new Fingerprint();
    result.fingerprint(fingerprint);
    assertThat(fingerprint.hexDigestAndReset()).isEqualTo(Fingerprint.getHexDigest(data));

    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    result.writeTo(bytesOut);
    assertThat(new String(bytesOut.toByteArray(), StandardCharsets.UTF_8)).isEqualTo(data);
  }

  private static void runSingleByteWriteTest(
      String data, boolean compressionEnabled, Class<? extends GenQueryResult> resultClass)
      throws IOException {
    GenQueryOutputStream underTest = new GenQueryOutputStream(compressionEnabled);
    for (byte b : data.getBytes(StandardCharsets.UTF_8)) {
      underTest.write(b);
    }
    underTest.close();

    GenQueryOutputStream.GenQueryResult result = underTest.getResult();
    assertThat(result).isInstanceOf(resultClass);
    assertThat(result.getBytes()).isEqualTo(ByteString.copyFromUtf8(data));
    assertThat(result.size()).isEqualTo(data.length());

    Fingerprint fingerprint = new Fingerprint();
    result.fingerprint(fingerprint);
    assertThat(fingerprint.hexDigestAndReset()).isEqualTo(Fingerprint.getHexDigest(data));

    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    result.writeTo(bytesOut);
    assertThat(new String(bytesOut.toByteArray(), StandardCharsets.UTF_8)).isEqualTo(data);
  }
}
