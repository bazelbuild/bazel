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
    for (String name : new String[] {"sha-256", "SHA-256", "SHA256", "sha256"}) {
      assertThat(converter.convert(name)).isSameInstanceAs(DigestHashFunction.SHA256);
    }
    for (String name : new String[] {"sha-1", "SHA-1", "SHA1", "sha1"}) {
      assertThat(converter.convert(name)).isSameInstanceAs(DigestHashFunction.SHA1);
    }
    for (String name : new String[] {"sha-512", "SHA-512", "SHA512", "sha512"}) {
      assertThat(converter.convert(name)).isSameInstanceAs(DigestHashFunction.SHA512);
    }
    for (String name : new String[] {"sha-384", "SHA-384", "SHA384", "sha384"}) {
      assertThat(converter.convert(name)).isSameInstanceAs(DigestHashFunction.SHA384);
    }
  }

  @Test
  public void lateRegistrationWithAlternativeNamesGetsPickedUpByConverter() throws Exception {
    DigestHashFunction.register(Hashing.goodFastHash(64), "MD5", "MD-5", "MD_5");

    assertThat(converter.convert("MD-5")).isSameInstanceAs(converter.convert("MD-5"));
    assertThat(converter.convert("Md-5")).isSameInstanceAs(converter.convert("MD-5"));
    assertThat(converter.convert("md-5")).isSameInstanceAs(converter.convert("MD-5"));

    assertThat(converter.convert("MD5")).isSameInstanceAs(converter.convert("MD-5"));
    assertThat(converter.convert("Md5")).isSameInstanceAs(converter.convert("MD-5"));
    assertThat(converter.convert("md5")).isSameInstanceAs(converter.convert("MD-5"));

    assertThat(converter.convert("MD_5")).isSameInstanceAs(converter.convert("MD-5"));
    assertThat(converter.convert("Md_5")).isSameInstanceAs(converter.convert("MD-5"));
    assertThat(converter.convert("md_5")).isSameInstanceAs(converter.convert("MD-5"));
  }
}
