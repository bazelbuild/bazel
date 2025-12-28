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
public class DigestHashFunctionGlobalsTest {
  private final DigestFunctionConverter converter = new DigestFunctionConverter();

  @Test
  public void convertReturnsTheSameValueAsTheConstant() throws Exception {
    assertThat(converter.convert("sha-256")).isSameInstanceAs(DigestHashFunction.SHA256);
    assertThat(converter.convert("SHA-256")).isSameInstanceAs(DigestHashFunction.SHA256);
    assertThat(converter.convert("SHA256")).isSameInstanceAs(DigestHashFunction.SHA256);
    assertThat(converter.convert("sha256")).isSameInstanceAs(DigestHashFunction.SHA256);

    assertThat(converter.convert("SHA-1")).isSameInstanceAs(DigestHashFunction.SHA1);
    assertThat(converter.convert("sha-1")).isSameInstanceAs(DigestHashFunction.SHA1);
    assertThat(converter.convert("SHA1")).isSameInstanceAs(DigestHashFunction.SHA1);
    assertThat(converter.convert("sha1")).isSameInstanceAs(DigestHashFunction.SHA1);
  }

  @Test
  public void lateRegistrationGetsPickedUpByConverter() throws Exception {
    DigestHashFunction.register(Hashing.goodFastHash(32), "MD5");

    assertThat(converter.convert("MD5")).isSameInstanceAs(converter.convert("md5"));
  }

  @Test
  public void lateRegistrationWithAlternativeNamesGetsPickedUpByConverter() throws Exception {
    DigestHashFunction.register(Hashing.goodFastHash(64), "SHA-224", "SHA224", "SHA_224");

    assertThat(converter.convert("SHA-224")).isSameInstanceAs(converter.convert("SHA-224"));
    assertThat(converter.convert("Sha-224")).isSameInstanceAs(converter.convert("SHA-224"));
    assertThat(converter.convert("sha-224")).isSameInstanceAs(converter.convert("SHA-224"));

    assertThat(converter.convert("SHA224")).isSameInstanceAs(converter.convert("SHA-224"));
    assertThat(converter.convert("Sha224")).isSameInstanceAs(converter.convert("SHA-224"));
    assertThat(converter.convert("sha224")).isSameInstanceAs(converter.convert("SHA-224"));

    assertThat(converter.convert("SHA_224")).isSameInstanceAs(converter.convert("SHA-224"));
    assertThat(converter.convert("Sha_224")).isSameInstanceAs(converter.convert("SHA-224"));
    assertThat(converter.convert("sha_224")).isSameInstanceAs(converter.convert("SHA-224"));
  }
}
