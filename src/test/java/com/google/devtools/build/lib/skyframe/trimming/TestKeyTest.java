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

package com.google.devtools.build.lib.skyframe.trimming;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for TestKey's parsing functionality. */
@RunWith(JUnit4.class)
public final class TestKeyTest {

  @Test
  public void parseEmptyConfig() throws Exception {
    assertThat(TestKey.parse("<> //foo").configuration()).isEmpty();
    assertThat(TestKey.parse("<    > //foo").configuration()).isEmpty();
    assertThat(TestKey.parse("  <    >    //foo").configuration()).isEmpty();
  }

  @Test
  public void parseOneElementConfig() throws Exception {
    assertThat(TestKey.parse("<A:1> //foo").configuration()).containsExactly("A", "1");
    assertThat(TestKey.parse("< A : 1 > //foo").configuration()).containsExactly("A", "1");
    assertThat(TestKey.parse("  <    A    :     1  >    //foo").configuration())
        .containsExactly("A", "1");
  }

  @Test
  public void parseMultiElementConfig() throws Exception {
    assertThat(TestKey.parse("<A:1,B:6,C:90> //foo").configuration())
        .containsExactly("A", "1", "B", "6", "C", "90");
    assertThat(TestKey.parse("< A : 1 , B : 6 , C : 90 > //foo").configuration())
        .containsExactly("A", "1", "B", "6", "C", "90");
    assertThat(TestKey.parse("  <    A    :     1, B: 6, C   :90  >    //foo").configuration())
        .containsExactly("A", "1", "B", "6", "C", "90");
  }

  @Test
  public void parseConfigWithSpaces() throws Exception {
    assertThat(TestKey.parse("<An Item: A Value> //foo").configuration())
        .containsExactly("An Item", "A Value");
  }

  @Test
  public void parseDescriptor() throws Exception {
    assertThat(TestKey.parse("<>//foo").descriptor()).isEqualTo("//foo");
    assertThat(TestKey.parse("<>    //foo   ").descriptor()).isEqualTo("//foo");
    assertThat(TestKey.parse("  <    A    :     1, B: 6, C   :90  >    //foo  ").descriptor())
        .isEqualTo("//foo");
  }

  @Test
  public void parseDescriptorWithSpaces() throws Exception {
    assertThat(TestKey.parse("<>//foo with space").descriptor()).isEqualTo("//foo with space");
    assertThat(TestKey.parse("<>    //foo  with   space ").descriptor())
        .isEqualTo("//foo  with   space");
  }

  @Test
  public void parseMissingConfiguration() throws Exception {
    assertThat(TestKey.parse("<>//foo with space").descriptor()).isEqualTo("//foo with space");
    assertThat(TestKey.parse("<>    //foo  with   space ").descriptor())
        .isEqualTo("//foo  with   space");
  }

  @Test
  public void equality() throws Exception {
    new EqualsTester()
        .addEqualityGroup(
            TestKey.parse("<>//foo"),
            TestKey.parse("  <   >   //foo   "),
            TestKey.create("//foo", ImmutableMap.of()))
        .addEqualityGroup(
            TestKey.parse("<A:1>//foo"),
            TestKey.parse("   <  A  :  1  >   //foo    "),
            TestKey.create("//foo", ImmutableMap.of("A", "1")))
        .addEqualityGroup(
            TestKey.parse("<>//bar"),
            TestKey.parse("   <   >   //bar    "),
            TestKey.create("//bar", ImmutableMap.of()))
        .addEqualityGroup(
            TestKey.parse("<A:1>//bar"),
            TestKey.parse("   < A : 1  >   //bar    "),
            TestKey.create("//bar", ImmutableMap.of("A", "1")))
        .testEquals();
  }

  @Test
  public void compareConfigurations_EqualCases() throws Exception {
    assertThat(TestKey.compareConfigurations(ImmutableMap.of(), ImmutableMap.of()))
        .isEqualTo(ConfigurationComparer.Result.EQUAL);
    assertThat(TestKey.compareConfigurations(ImmutableMap.of("A", "1"), ImmutableMap.of("A", "1")))
        .isEqualTo(ConfigurationComparer.Result.EQUAL);
  }

  @Test
  public void compareConfigurations_SubsetCases() throws Exception {
    assertThat(TestKey.compareConfigurations(ImmutableMap.of(), ImmutableMap.of("A", "1")))
        .isEqualTo(ConfigurationComparer.Result.SUBSET);
    assertThat(
            TestKey.compareConfigurations(
                ImmutableMap.of("A", "1"), ImmutableMap.of("A", "1", "B", "2")))
        .isEqualTo(ConfigurationComparer.Result.SUBSET);
  }

  @Test
  public void compareConfigurations_SupersetCases() throws Exception {
    assertThat(TestKey.compareConfigurations(ImmutableMap.of("A", "1"), ImmutableMap.of()))
        .isEqualTo(ConfigurationComparer.Result.SUPERSET);
    assertThat(
            TestKey.compareConfigurations(
                ImmutableMap.of("A", "1", "B", "2"), ImmutableMap.of("A", "1")))
        .isEqualTo(ConfigurationComparer.Result.SUPERSET);
  }

  @Test
  public void compareConfigurations_AllSharedFragmentsEqualCases() throws Exception {
    assertThat(TestKey.compareConfigurations(ImmutableMap.of("A", "1"), ImmutableMap.of("B", "1")))
        .isEqualTo(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL);
    assertThat(
            TestKey.compareConfigurations(
                ImmutableMap.of("A", "1", "B", "2"), ImmutableMap.of("A", "1", "C", "3")))
        .isEqualTo(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL);
  }

  @Test
  public void compareConfigurations_DifferentCases() throws Exception {
    assertThat(TestKey.compareConfigurations(ImmutableMap.of("A", "1"), ImmutableMap.of("A", "2")))
        .isEqualTo(ConfigurationComparer.Result.DIFFERENT);
    assertThat(
            TestKey.compareConfigurations(
                ImmutableMap.of("A", "1", "B", "2", "C", "3"),
                ImmutableMap.of("A", "2", "B", "2", "D", "4")))
        .isEqualTo(ConfigurationComparer.Result.DIFFERENT);
  }
}
