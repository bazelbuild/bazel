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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for String canonicalizer.
 */
@RunWith(JUnit4.class)
public class StringCanonicalizerTest {

  @Test
  public void twoDifferentStringsAreDifferent() {
    String stringA = StringCanonicalizer.intern("A");
    String stringB = StringCanonicalizer.intern("B");
    assertThat(stringA).isNotEqualTo(stringB);
  }

  @Test
  public void twoSameStringsAreCanonicalized() {
    String stringA1 = StringCanonicalizer.intern(new String("A"));
    String stringA2 = StringCanonicalizer.intern(new String("A"));
    assertThat(stringA2).isSameInstanceAs(stringA1);
  }
}
