// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.hash;

import static org.junit.Assert.assertEquals;

import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Blake3Hasher}. */
@RunWith(JUnit4.class)
public class Blake3HasherTest {
  private static String hexString(byte[] byteArray) {
    StringBuilder stringBuilder = new StringBuilder(byteArray.length * 2);
    for (byte b : byteArray) {
      stringBuilder.append(String.format("%02x", b));
    }
    return stringBuilder.toString();
  }

  @Test
  public void emptyHash() {
    Blake3Hasher h = new Blake3Hasher();

    byte[] data = new byte[0];
    h.update(data);
    byte[] output = h.getOutput(Blake3Hasher.OUT_LEN);

    assertEquals(
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262", hexString(output));
  }

  @Test
  public void helloWorld() {
    Blake3Hasher h = new Blake3Hasher();

    byte[] data = "hello world".getBytes(StandardCharsets.US_ASCII);
    h.update(data);
    byte[] output = h.getOutput(Blake3Hasher.OUT_LEN);

    assertEquals(
        "d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24", hexString(output));
  }
}
