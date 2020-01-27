// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.testing.NullPointerTester;
import com.google.devtools.build.lib.analysis.actions.LauncherFileWriteAction.LaunchInfo;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link LauncherFileWriteAction}. */
@RunWith(JUnit4.class)
public class LauncherFileWriteActionTest {

  @Test
  public void testAddKeyValuePair() throws Exception {
    LaunchInfo actual =
        LaunchInfo.builder()
            .addKeyValuePair("", "won't show up")
            .addKeyValuePair("foo", "bar")
            .addKeyValuePair("baz", null)
            .build();
    ByteArrayOutputStream expected = new ByteArrayOutputStream();
    expected.write("foo=bar\0".getBytes(StandardCharsets.UTF_8));
    expected.write("baz=\0".getBytes(StandardCharsets.UTF_8));
    assertOutput(actual, expected.toByteArray());
  }

  @Test
  public void testKeyValueFingerprint() throws Exception {
    // LaunchInfos with different entries should have different fingerprints.
    assertThat(LaunchInfo.builder().addKeyValuePair("foo", "bar").build().fingerPrint)
        .isNotEqualTo(LaunchInfo.builder().addKeyValuePair("bar", "foo").build().fingerPrint);

    // LaunchInfos with the same entries but in different order should have different fingerprints.
    assertThat(
            LaunchInfo.builder()
                .addKeyValuePair("foo", "bar")
                .addKeyValuePair("bar", "foo")
                .build()
                .fingerPrint)
        .isNotEqualTo(
            LaunchInfo.builder()
                .addKeyValuePair("bar", "foo")
                .addKeyValuePair("foo", "bar")
                .build()
                .fingerPrint);

    // Two identically-constructed LaunchInfos should have the same fingerprint.
    assertThat(
            LaunchInfo.builder()
                .addKeyValuePair("foo", "bar")
                .addKeyValuePair("bar", "foo")
                .build()
                .fingerPrint)
        .isEqualTo(
            LaunchInfo.builder()
                .addKeyValuePair("foo", "bar")
                .addKeyValuePair("bar", "foo")
                .build()
                .fingerPrint);
  }

  @Test
  public void testAddJoinedValues() throws Exception {
    LaunchInfo actual =
        LaunchInfo.builder()
            .addJoinedValues("foo", "", ImmutableList.of())
            .addJoinedValues("bar", "x", ImmutableList.of())
            .addJoinedValues("baz", ";", ImmutableList.of("aa"))
            .addJoinedValues("qux", ":", ImmutableList.of("aa", "bb", "cc"))
            .addJoinedValues("mex", "--", ImmutableList.of("aa", "bb", "cc"))
            .build();
    ByteArrayOutputStream expected = new ByteArrayOutputStream();
    expected.write("foo=\0".getBytes(StandardCharsets.UTF_8));
    expected.write("bar=\0".getBytes(StandardCharsets.UTF_8));
    expected.write("baz=aa\0".getBytes(StandardCharsets.UTF_8));
    expected.write("qux=aa:bb:cc\0".getBytes(StandardCharsets.UTF_8));
    expected.write("mex=aa--bb--cc\0".getBytes(StandardCharsets.UTF_8));
    assertOutput(actual, expected.toByteArray());
  }

  @Test
  public void testJoinedValuesFingerprint() throws Exception {
    // LaunchInfos with different entries should have different fingerprints.
    assertThat(
            LaunchInfo.builder()
                .addJoinedValues("foo", ";", ImmutableList.of("aa", "bb"))
                .build()
                .fingerPrint)
        .isNotEqualTo(
            LaunchInfo.builder()
                .addJoinedValues("bar", ";", ImmutableList.of("aa", "bb"))
                .build()
                .fingerPrint);

    // LaunchInfos with the same entries but in different order should have different fingerprints.
    assertThat(
            LaunchInfo.builder()
                .addJoinedValues("foo", ";", ImmutableList.of("aa", "bb"))
                .addJoinedValues("bar", ";", ImmutableList.of("aa", "bb"))
                .build()
                .fingerPrint)
        .isNotEqualTo(
            LaunchInfo.builder()
                .addJoinedValues("bar", ";", ImmutableList.of("aa", "bb"))
                .addJoinedValues("foo", ";", ImmutableList.of("aa", "bb"))
                .build()
                .fingerPrint);

    // Two identically-constructed LaunchInfos should have the same fingerprint.
    assertThat(
            LaunchInfo.builder()
                .addJoinedValues("foo", ";", ImmutableList.of("aa", "bb"))
                .addJoinedValues("bar", ";", ImmutableList.of("aa", "bb"))
                .build()
                .fingerPrint)
        .isEqualTo(
            LaunchInfo.builder()
                .addJoinedValues("foo", ";", ImmutableList.of("aa", "bb"))
                .addJoinedValues("bar", ";", ImmutableList.of("aa", "bb"))
                .build()
                .fingerPrint);
  }

  @Test
  public void testFingerprintDependsOnEntryType() throws Exception {
    // Although these LaunchInfo objects render to the same octet stream, their fingerprint is
    // different because we construct them differently.
    LaunchInfo actual1 = LaunchInfo.builder().addKeyValuePair("foo", "bar;baz").build();
    LaunchInfo actual2 =
        LaunchInfo.builder().addJoinedValues("foo", ";", ImmutableList.of("bar", "baz")).build();
    try (ByteArrayOutputStream out1 = new ByteArrayOutputStream();
        ByteArrayOutputStream out2 = new ByteArrayOutputStream()) {
      actual1.write(out1);
      actual2.write(out2);
      assertThat(out1.toByteArray()).isEqualTo(out2.toByteArray());
    }
    assertThat(actual1.fingerPrint).isNotEqualTo(actual2.fingerPrint);
  }

  @Test
  public void testNulls() throws Exception {
    assertOutput(LaunchInfo.builder().build(), new byte[0]);

    assertOutput(
        LaunchInfo.builder().addKeyValuePair("", null).addKeyValuePair("", "").build(),
        new byte[0]);

    assertOutput(
        LaunchInfo.builder()
            .addJoinedValues("", "", null)
            .addJoinedValues("", "delimiter", null)
            .addJoinedValues("", "", ImmutableList.of())
            .addJoinedValues("", "delimiter", ImmutableList.of())
            .build(),
        new byte[0]);

    LaunchInfo.Builder obj = LaunchInfo.builder();
    Class<LaunchInfo.Builder> clazz = LaunchInfo.Builder.class;
    NullPointerTester npt = new NullPointerTester().setDefault(String.class, "foo");

    npt.testMethod(obj, clazz.getMethod("addKeyValuePair", String.class, String.class));
    npt.testMethod(
        obj, clazz.getMethod("addJoinedValues", String.class, String.class, Iterable.class));
  }

  private static void assertOutput(LaunchInfo actual, byte[] expected) throws Exception {
    try (ByteArrayOutputStream out = new ByteArrayOutputStream()) {
      assertThat(actual.write(out)).isEqualTo(expected.length);
      assertThat(out.toByteArray()).isEqualTo(expected);
    }
  }
}
