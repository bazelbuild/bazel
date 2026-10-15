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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for Fingerprint.
 */
@RunWith(JUnit4.class)
public class FingerprintTest {

  private static void assertFingerprintsDiffer(List<String> list1, List<String>list2) {
    Fingerprint f1 = new Fingerprint();
    Fingerprint f1Latin1 = new Fingerprint();
    for (String s : list1) {
      f1.addString(s);
      f1Latin1.addString(s);
    }
    Fingerprint f2 = new Fingerprint();
    Fingerprint f2Latin1 = new Fingerprint();
    for (String s : list2) {
      f2.addString(s);
      f2Latin1.addString(s);
    }
    assertThat(f1.hexDigestAndReset()).isNotEqualTo(f2.hexDigestAndReset());
    assertThat(f1Latin1.hexDigestAndReset()).isNotEqualTo(f2Latin1.hexDigestAndReset());
  }

  @Test
  public void equivalentBytesAndStringsFingerprintsMatch() {
    String helloWorld = "Hello World!";
    // $ echo -n 'Hello World!' | sha256sum
    String helloWorldHash = "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069";

    assertThat(new Fingerprint().addBytes(helloWorld.getBytes(UTF_8)).hexDigestAndReset())
        .isEqualTo(helloWorldHash);

    assertThat(Fingerprint.getHexDigest(helloWorld)).isEqualTo(helloWorldHash);

    assertThat(new Fingerprint().addBytes(ByteString.copyFromUtf8(helloWorld)).hexDigestAndReset())
        .isEqualTo(helloWorldHash);
  }

  @Test
  public void stringEncodingUsesLengthPrefixedInternalBytes() throws Exception {
    byte[] allBytes = new byte[256];
    for (int i = 0; i < allBytes.length; i++) {
      allBytes[i] = (byte) i;
    }
    Fingerprint fingerprint = new Fingerprint();
    for (String input :
        ImmutableList.of(
            "",
            "hello",
            "a".repeat(1024),
            "a".repeat(10000),
            new String(allBytes, ISO_8859_1),
            new String(allBytes, ISO_8859_1).repeat(100),
            new String("é中😀".getBytes(UTF_8), ISO_8859_1))) {
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      CodedOutputStream protobuf = CodedOutputStream.newInstance(bytes, 1024);
      protobuf.writeInt32NoTag(-123);
      protobuf.writeByteArrayNoTag(input.getBytes(ISO_8859_1));
      protobuf.writeBoolNoTag(true);
      protobuf.writeStringNoTag("suffix");
      protobuf.flush();
      byte[] expected = DigestHashFunction.SHA256.newMessageDigest().digest(bytes.toByteArray());
      assertThat(
              fingerprint
                  .addInt(-123)
                  .addString(input)
                  .addBoolean(true)
                  .addString("suffix")
                  .digestAndReset())
          .isEqualTo(expected);
    }
  }

  @Test
  public void stringEncodingRejectsNonInternalStrings() {
    assertThrows(IllegalArgumentException.class, () -> new Fingerprint().addString("中"));
  }

  @Test
  public void otherStringFingerprint() {
    assertFingerprintsDiffer(ImmutableList.of("Hello World!"), ImmutableList.of("Goodbye World."));
  }

  @Test
  public void multipleUpdatesDiffer() throws Exception {
    assertFingerprintsDiffer(ImmutableList.of("Hello ", "World!"),
                             ImmutableList.of("Hello World!"));
  }

  @Test
  public void multipleUpdatesShiftedDiffer() throws Exception {
    assertFingerprintsDiffer(ImmutableList.of("Hello ", "World!"),
                             ImmutableList.of("Hello", " World!"));
  }

  @Test
  public void listFingerprintNotSameAsIndividualElements() throws Exception {
    Fingerprint f1 = new Fingerprint();
    f1.addString("Hello ");
    f1.addString("World!");
    Fingerprint f2 = new Fingerprint();
    f2.addStrings(ImmutableList.of("Hello ", "World!"));
    assertThat(f1.hexDigestAndReset()).isNotEqualTo(f2.hexDigestAndReset());
  }

  @Test
  public void mapFingerprintNotSameAsIndividualElements() throws Exception {
    Fingerprint f1 = new Fingerprint();
    Map<String, String> map = new HashMap<>();
    map.put("Hello ", "World!");
    f1.addStringMap(map);
    Fingerprint f2 = new Fingerprint();
    f2.addStrings(ImmutableList.of("Hello ", "World!"));
    assertThat(f1.hexDigestAndReset()).isNotEqualTo(f2.hexDigestAndReset());
  }

  @Test
  public void addBoolean() throws Exception {
    String f1 = new Fingerprint().addBoolean(true).hexDigestAndReset();
    String f2 = new Fingerprint().addBoolean(false).hexDigestAndReset();
    String f3 = new Fingerprint().addBoolean(true).hexDigestAndReset();

    assertThat(f1).isEqualTo(f3);
    assertThat(f1).isNotEqualTo(f2);
  }

  @Test
  public void addPath() throws Exception {
    PathFragment pf = PathFragment.create("/etc/pwd");
    assertThat(new Fingerprint().addPath(pf).hexDigestAndReset())
        .isEqualTo("0b229115c2da46773ff38528420b922488dd564ddb3c0c861fb1c77ae8525f9b");
    Path p = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256).getPath(pf);
    assertThat(new Fingerprint().addPath(p).hexDigestAndReset())
        .isEqualTo("0b229115c2da46773ff38528420b922488dd564ddb3c0c861fb1c77ae8525f9b");
  }

  @Test
  public void addNullableBoolean() throws Exception {
    String f1 = new Fingerprint().addNullableBoolean(null).hexDigestAndReset();
    assertThat(f1).isEqualTo(new Fingerprint().addNullableBoolean(null).hexDigestAndReset());
    assertThat(f1).isNotEqualTo(new Fingerprint().addNullableBoolean(false).hexDigestAndReset());
    assertThat(f1).isNotEqualTo(new Fingerprint().addNullableBoolean(true).hexDigestAndReset());
  }

  @Test
  public void addNullableInteger() throws Exception {
    String f1 = new Fingerprint().addNullableInt(null).hexDigestAndReset();
    assertThat(f1).isEqualTo(new Fingerprint().addNullableInt(null).hexDigestAndReset());
    assertThat(f1).isNotEqualTo(new Fingerprint().addNullableInt(0).hexDigestAndReset());
    assertThat(f1).isNotEqualTo(new Fingerprint().addNullableInt(1).hexDigestAndReset());
  }

  @Test
  public void addNullableString() throws Exception {
    String f1 = new Fingerprint().addNullableString(null).hexDigestAndReset();
    assertThat(f1).isEqualTo(new Fingerprint().addNullableString(null).hexDigestAndReset());
    assertThat(f1).isNotEqualTo(new Fingerprint().addNullableString("").hexDigestAndReset());
  }

  @Test
  public void testReusableAfterReset() throws Exception {
    Fingerprint fp = new Fingerprint();
    String f1 = convolutedFingerprintAndReset(fp);
    String f2 = convolutedFingerprintAndReset(fp);
    assertThat(f1).isEqualTo(f2);
  }

  private static String convolutedFingerprintAndReset(Fingerprint fingerprint) {
    return fingerprint
        .addBoolean(false)
        .addBytes(new byte[10])
        .addBytes(new byte[10], 0, 5)
        .addInt(20)
        .addLong(30)
        .addNullableBoolean(null)
        .addNullableInt(null)
        .addNullableString(null)
        .addPath(PathFragment.create("/foo/bar"))
        .addPaths(ImmutableList.of(PathFragment.create("/foo/bar")))
        .addString("baz")
        .addUUID(UUID.fromString("12345678-1234-1234-1234-1234567890ab"))
        .hexDigestAndReset();
  }
}
