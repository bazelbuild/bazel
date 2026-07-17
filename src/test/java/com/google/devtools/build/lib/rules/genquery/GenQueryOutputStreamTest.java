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

import com.google.devtools.build.lib.rules.genquery.GenQueryOutputStream.GenQueryResult;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.protobuf.ByteString;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.zip.GZIPOutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link GenQueryOutputStream}. */
@RunWith(TestParameterInjector.class)
public class GenQueryOutputStreamTest {

  @TestParameter private boolean outputCompressed;

  @Test
  public void testSmallOutputMultibyteWrite() throws IOException {
    runMultibyteWriteTest(
        "xyz".repeat(10_000), GenQueryOutputStream.SimpleResult.class, outputCompressed);
  }

  @Test
  public void testBigOutputMultibyteWrite() throws IOException {
    runMultibyteWriteTest(
        "xyz".repeat(1_000_000),
        outputCompressed
            ? GenQueryOutputStream.SimpleResult.class
            : GenQueryOutputStream.CompressedResultWithDecompressedOutput.class,
        outputCompressed);
  }

  @Test
  public void testSmallOutputSingleByteWrites() throws IOException {
    runSingleByteWriteTest(
        "xyz".repeat(10_000), GenQueryOutputStream.SimpleResult.class, outputCompressed);
  }

  @Test
  public void testBigOutputSingleByteWrites() throws IOException {
    runSingleByteWriteTest(
        "xyz".repeat(1_000_000),
        outputCompressed
            ? GenQueryOutputStream.SimpleResult.class
            : GenQueryOutputStream.CompressedResultWithDecompressedOutput.class,
        outputCompressed);
  }

  private static void runMultibyteWriteTest(
      String data, Class<? extends GenQueryResult> resultClass, boolean outputCompressed)
      throws IOException {
    GenQueryOutputStream underTest = new GenQueryOutputStream(outputCompressed);
    underTest.write(data.getBytes(StandardCharsets.UTF_8));
    underTest.close();

    verifyGenQueryResult(underTest.getResult(), data, resultClass, outputCompressed);
  }

  private static void runSingleByteWriteTest(
      String data, Class<? extends GenQueryResult> resultClass, boolean outputCompressed)
      throws IOException {
    GenQueryOutputStream underTest = new GenQueryOutputStream(outputCompressed);
    for (byte b : data.getBytes(StandardCharsets.UTF_8)) {
      underTest.write(b);
    }
    underTest.close();

    verifyGenQueryResult(underTest.getResult(), data, resultClass, outputCompressed);
  }

  private static void verifyGenQueryResult(
      GenQueryOutputStream.GenQueryResult result,
      String data,
      Class<? extends GenQueryResult> resultClass,
      boolean outputCompressed)
      throws IOException {
    assertThat(result).isInstanceOf(resultClass);

    if (outputCompressed) {
      // If result is actually compressed, also compress input data so that it is comparable to what
      // is outputted from GenQueryResult.
      ByteString dataInByteString = ByteString.copyFromUtf8(data);
      ByteString.Output compressedDataBytesOut = ByteString.newOutput();
      GZIPOutputStream gzipDataOut = new GZIPOutputStream(compressedDataBytesOut);
      dataInByteString.writeTo(gzipDataOut);
      gzipDataOut.finish();
      ByteString dataCompressedInByteString = compressedDataBytesOut.toByteString();

      assertThat(result.getBytes()).isEqualTo(dataCompressedInByteString);

      Fingerprint actualFingerprint = new Fingerprint();
      result.fingerprint(actualFingerprint);
      Fingerprint expectFingerprint = new Fingerprint();
      expectFingerprint.addBytes(dataCompressedInByteString);
      assertThat(actualFingerprint.hexDigestAndReset())
          .isEqualTo(expectFingerprint.hexDigestAndReset());

      ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
      result.writeTo(bytesOut);
      assertThat(bytesOut.toByteArray()).isEqualTo(dataCompressedInByteString.toByteArray());
    } else {
      assertThat(result.getBytes()).isEqualTo(ByteString.copyFromUtf8(data));
      assertThat(result.size()).isEqualTo(data.length());

      Fingerprint actualFingerprint = new Fingerprint();
      result.fingerprint(actualFingerprint);
      Fingerprint expectFingerprint = new Fingerprint();
      expectFingerprint.addBytes(data.getBytes(StandardCharsets.UTF_8));
      assertThat(actualFingerprint.hexDigestAndReset())
          .isEqualTo(expectFingerprint.hexDigestAndReset());

      ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
      result.writeTo(bytesOut);
      assertThat(new String(bytesOut.toByteArray(), StandardCharsets.UTF_8)).isEqualTo(data);
    }
  }
}
