// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.vfs.DigestHashFunction.DigestFunctionConverter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for DigestHashFunction, notably that the static instances can be compared with reference
 * equality.
 */
@RunWith(JUnit4.class)
public class DigestHashFunctionTest {
  private final DigestFunctionConverter converter = new DigestFunctionConverter();

  @Test
  public void convertReturnsTheSameValueAsTheConstant() throws Exception {
    assertThat(converter.convert("sha-256")).isSameAs(DigestHashFunction.SHA256);
    assertThat(converter.convert("SHA-256")).isSameAs(DigestHashFunction.SHA256);
    assertThat(converter.convert("SHA256")).isSameAs(DigestHashFunction.SHA256);
    assertThat(converter.convert("sha256")).isSameAs(DigestHashFunction.SHA256);

    assertThat(converter.convert("SHA-1")).isSameAs(DigestHashFunction.SHA1);
    assertThat(converter.convert("sha-1")).isSameAs(DigestHashFunction.SHA1);
    assertThat(converter.convert("SHA1")).isSameAs(DigestHashFunction.SHA1);
    assertThat(converter.convert("sha1")).isSameAs(DigestHashFunction.SHA1);

    assertThat(converter.convert("MD5")).isSameAs(DigestHashFunction.MD5);
    assertThat(converter.convert("md5")).isSameAs(DigestHashFunction.MD5);
  }

  @Test
  public void lateRegistrationGetsPickedUpByConverter() throws Exception {
    DigestHashFunction.register(Hashing.goodFastHash(32), "goodFastHash32");

    assertThat(converter.convert("goodFastHash32")).isSameAs(converter.convert("GOODFASTHASH32"));
  }

  @Test
  public void lateRegistrationWithAlternativeNamesGetsPickedUpByConverter() throws Exception {
    DigestHashFunction.register(
        Hashing.goodFastHash(64), "goodFastHash64", "goodFastHash-64", "good-fast-hash-64");

    assertThat(converter.convert("goodFastHash64")).isSameAs(converter.convert("GOODFASTHASH64"));
    assertThat(converter.convert("goodFastHash64")).isSameAs(converter.convert("goodFastHash-64"));
    assertThat(converter.convert("goodFastHash64"))
        .isSameAs(converter.convert("good-fast-hash-64"));
    assertThat(converter.convert("goodFastHash64"))
        .isSameAs(converter.convert("GOOD-fast-HASH-64"));
  }
}
