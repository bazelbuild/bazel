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
  public void valid() {
    Blake3Hasher h = new Blake3Hasher();
    assertEquals(true, h.isValid());
    h.close();
    assertEquals(false, h.isValid());
  }

  @Test
  public void doubleClose() {
    Blake3Hasher h = new Blake3Hasher();
    assertEquals(true, h.isValid());
    h.close();
    h.close();
    assertEquals(false, h.isValid());
  }

  @Test
  public void initKeyed() {
    Blake3Hasher h = new Blake3Hasher();
    byte[] key = "sothisisarandomstring32byteslong".getBytes(StandardCharsets.US_ASCII);
    h.initKeyed(key);
    assertEquals(true, h.isValid());

    byte[] data = new byte[0];
    h.update(data);
    byte[] output = h.getOutput();
    h.close();

    assertEquals(
        "57c3c9928a850ea5b94ea8fbe6b2283774a9e52918ddf0471997da7272a017f2", hexString(output));
  }

  @Test
  public void initDeriveKey() {
    Blake3Hasher h = new Blake3Hasher();
    h.initDeriveKey("sothisisarandomstring32byteslong");
    assertEquals(true, h.isValid());

    byte[] data = new byte[0];
    h.update(data);
    byte[] output = h.getOutput();
    h.close();

    assertEquals(
        "3189ab35d0798ed6368fdf3dea41a8e047638d2974f6a455c1df0314a4e7794e", hexString(output));
  }

  @Test
  public void emptyHash() {
    Blake3Hasher h = new Blake3Hasher();
    assertEquals(true, h.isValid());

    byte[] data = new byte[0];
    h.update(data);
    byte[] output = h.getOutput();
    h.close();

    assertEquals(
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262", hexString(output));
  }

  @Test
  public void helloWorld() {
    Blake3Hasher h = new Blake3Hasher();
    assertEquals(true, h.isValid());

    byte[] data = "hello world".getBytes(StandardCharsets.US_ASCII);
    h.update(data);
    byte[] output = h.getOutput();
    h.close();

    assertEquals(
        "d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24", hexString(output));
  }
}
