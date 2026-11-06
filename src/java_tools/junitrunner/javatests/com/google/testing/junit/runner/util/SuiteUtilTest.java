// Copyright 2024 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.util;

import static com.google.common.truth.Truth.assertThat;

import java.util.Arrays;
import java.util.Collections;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SuiteUtil}. */
@RunWith(JUnit4.class)
@SuppressWarnings("JdkImmutableCollections") // avoiding dep on com.google.common.
public class SuiteUtilTest {

  @Test
  public void testParseSuiteClassNames_nullOrBlank() {
    assertThat(SuiteUtil.parseSuiteClassNames(null)).isEmpty();
    assertThat(SuiteUtil.parseSuiteClassNames("")).isEmpty();
    assertThat(SuiteUtil.parseSuiteClassNames("   ")).isEmpty();
  }

  @Test
  public void testParseSuiteClassNames_singleClass() {
    assertThat(SuiteUtil.parseSuiteClassNames("com.example.FooTest"))
        .containsExactly("com.example.FooTest");
  }

  @Test
  public void testParseSuiteClassNames_multipleClassesTrimmingAndDeduplication() {
    assertThat(
            SuiteUtil.parseSuiteClassNames(
                "  com.example.FooTest , com.example.BarTest, com.example.FooTest ,"
                    + " com.example.BazTest "))
        .containsExactly("com.example.FooTest", "com.example.BarTest", "com.example.BazTest")
        .inOrder();
  }

  @Test
  public void testGetTopLevelSuiteName_singleClass() {
    assertThat(SuiteUtil.getTopLevelSuiteName(Collections.singletonList(String.class)))
        .isEqualTo(String.class.getCanonicalName());
  }

  @Test
  public void testGetTopLevelSuiteName_explicitTestTarget() {
    assertThat(
            SuiteUtil.getTopLevelSuiteName(
                Arrays.asList(String.class, Integer.class), "//foo:bar"))
        .isEqualTo("//foo:bar");
  }

  @Test
  public void testGetTopLevelSuiteName_multipleClassesFallbackToPackage() {
    assertThat(
            SuiteUtil.getTopLevelSuiteName(Arrays.asList(String.class, Integer.class), null))
        .isEqualTo("java.lang");
  }

  @Test
  public void testGetTopLevelSuiteName_emptyFallbackToDefault() {
    assertThat(SuiteUtil.getTopLevelSuiteName(Collections.emptyList(), null))
        .isEqualTo("TestSuite");
  }
}
