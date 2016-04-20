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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
      f1Latin1.addStringLatin1(s);
    }
    Fingerprint f2 = new Fingerprint();
    Fingerprint f2Latin1 = new Fingerprint();
    for (String s : list2) {
      f2.addString(s);
      f2Latin1.addStringLatin1(s);
    }
    assertThat(f1.hexDigestAndReset()).isNotEqualTo(f2.hexDigestAndReset());
    assertThat(f1Latin1.hexDigestAndReset()).isNotEqualTo(f2Latin1.hexDigestAndReset());
  }

  // You can validate the md5 of the simple string against
  // echo -n 'Hello World!'| md5sum
  @Test
  public void bytesFingerprint() {
    assertThat("ed076287532e86365e841e92bfc50d8c").isEqualTo(
        new Fingerprint().addBytes("Hello World!".getBytes(UTF_8)).hexDigestAndReset());
    assertThat("ed076287532e86365e841e92bfc50d8c").isEqualTo(Fingerprint.md5Digest("Hello World!"));
  }

  @Test
  public void otherStringFingerprint() {
    assertFingerprintsDiffer(ImmutableList.of("Hello World!"),
                             ImmutableList.of("Goodbye World."));
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
  public void toStringTest() throws Exception {
    Fingerprint f1 = new Fingerprint();
    f1.addString("Hello ");
    f1.addString("World!");
    String fp = f1.hexDigestAndReset();
    Fingerprint f2 = new Fingerprint();
    f2.addString("Hello ");
    // make sure that you can call toString on the intermediate result
    // and continue with the operation.
    assertThat(fp).isNotEqualTo(f2.toString());
    f2.addString("World!");
    assertThat(fp).isEqualTo(f2.hexDigestAndReset());
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
    PathFragment pf = new PathFragment("/etc/pwd");
    assertThat("01cc3eeea3a2f58e447e824f9f62d3d1").isEqualTo(
        new Fingerprint().addPath(pf).hexDigestAndReset());
    Path p = new InMemoryFileSystem(BlazeClock.instance()).getPath(pf);
    assertThat("01cc3eeea3a2f58e447e824f9f62d3d1").isEqualTo(
        new Fingerprint().addPath(p).hexDigestAndReset());
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
}
