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
package com.google.devtools.build.android;

import static com.google.common.truth.Truth.assertThat;

import com.android.builder.core.VariantType;
import com.android.repository.Revision;
import com.google.common.collect.ImmutableList;
import com.google.common.testing.NullPointerTester;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AaptCommandBuilder}. */
@RunWith(JUnit4.class)
public class AaptCommandBuilderTest {
  private Path aapt;
  private Path manifest;

  @Before
  public void createPaths() {
    FileSystem fs = FileSystems.getDefault();
    aapt = fs.getPath("aapt");
    manifest = fs.getPath("AndroidManifest.xml");
  }

  @Test
  public void testPassesNullPointerTester() throws Exception {
    NullPointerTester tester = new NullPointerTester().setDefault(Path.class, aapt);

    tester.testConstructor(AaptCommandBuilder.class.getConstructor(Path.class));
    tester.ignore(AaptCommandBuilder.class.getMethod("execute", String.class));
    tester.testAllPublicInstanceMethods(new AaptCommandBuilder(aapt));
    tester.testAllPublicInstanceMethods(new AaptCommandBuilder(aapt).when(true));
    tester.testAllPublicInstanceMethods(new AaptCommandBuilder(aapt).when(false));
  }

  @Test
  public void testAaptPathAddedAsFirstArgument() {
    assertThat(new AaptCommandBuilder(aapt).build()).containsExactly(aapt.toString());
  }

  @Test
  public void testAddArgumentAddedToList() {
    assertThat(new AaptCommandBuilder(aapt).add("package").build()).contains("package");
  }

  @Test
  public void testAddCallsAddedToEndOfList() {
    assertThat(new AaptCommandBuilder(aapt).add("package").add("-f").build())
        .containsExactly(aapt.toString(), "package", "-f")
        .inOrder();
  }

  @Test
  public void testAddWithStringValueAddsFlagThenValueAsSeparateWords() {
    assertThat(new AaptCommandBuilder(aapt).add("-0", "gif").build())
        .containsExactly(aapt.toString(), "-0", "gif")
        .inOrder();
  }

  @Test
  public void testAddWithEmptyValueAddsNothing() {
    assertThat(new AaptCommandBuilder(aapt).add("-0", "").build()).doesNotContain("-0");
  }

  @Test
  public void testAddWithNullStringValueAddsNothing() {
    assertThat(new AaptCommandBuilder(aapt).add("-0", (String) null).build()).doesNotContain("-0");
  }

  @Test
  public void testAddWithPathValueAddsFlagThenStringValueAsSeparateWords() {
    assertThat(new AaptCommandBuilder(aapt).add("-M", manifest).build())
        .containsExactly(aapt.toString(), "-M", manifest.toString())
        .inOrder();
  }

  @Test
  public void testAddWithNullPathValueAddsNothing() {
    assertThat(new AaptCommandBuilder(aapt).add("-M", (Path) null).build()).doesNotContain("-M");
  }

  @Test
  public void testAddRepeatedWithEmptyValuesAddsNothing() {
    assertThat(new AaptCommandBuilder(aapt).addRepeated("-0", ImmutableList.<String>of()).build())
        .doesNotContain("-0");
  }

  @Test
  public void testAddRepeatedWithSingleValueAddsOneFlagOneValue() {
    assertThat(new AaptCommandBuilder(aapt).addRepeated("-0", ImmutableList.of("gif")).build())
        .containsExactly(aapt.toString(), "-0", "gif")
        .inOrder();
  }

  @Test
  public void testAddRepeatedWithMultipleValuesAddsFlagBeforeEachValue() {
    assertThat(
            new AaptCommandBuilder(aapt).addRepeated("-0", ImmutableList.of("gif", "png")).build())
        .containsExactly(aapt.toString(), "-0", "gif", "-0", "png")
        .inOrder();
  }

  @Test
  public void testAddRepeatedSkipsNullValues() {
    ArrayList<String> list = new ArrayList<>(3);
    list.add("gif");
    list.add(null);
    list.add("png");
    assertThat(
            new AaptCommandBuilder(aapt).addRepeated("-0", list).build())
        .containsExactly(aapt.toString(), "-0", "gif", "-0", "png")
        .inOrder();
  }


  @Test
  public void testThenAddFlagForwardsCallAfterWhenTrue() {
    assertThat(
        new AaptCommandBuilder(aapt).when(true).thenAdd("--addthisflag").build())
        .contains("--addthisflag");
  }

  @Test
  public void testThenAddFlagWithValueForwardsCallAfterWhenTrue() {
    assertThat(
        new AaptCommandBuilder(aapt)
            .when(true).thenAdd("--addthisflag", "andthisvalue").build())
        .contains("--addthisflag");
  }

  @Test
  public void testThenAddFlagWithPathForwardsCallAfterWhenTrue() {
    assertThat(
        new AaptCommandBuilder(aapt)
            .when(true).thenAdd("--addthisflag", manifest).build())
        .contains("--addthisflag");
  }

  @Test
  public void testThenAddRepeatedForwardsCallAfterWhenTrue() {
    assertThat(
        new AaptCommandBuilder(aapt)
            .when(true).thenAddRepeated("--addthisflag", ImmutableList.of("andthesevalues"))
            .build())
        .contains("--addthisflag");
  }

  @Test
  public void testThenAddFlagDoesNothingAfterWhenFalse() {
    assertThat(
        new AaptCommandBuilder(aapt).when(false).thenAdd("--dontaddthisflag").build())
        .doesNotContain("--dontaddthisflag");
  }

  @Test
  public void testThenAddFlagWithValueDoesNothingAfterWhenFalse() {
    assertThat(
        new AaptCommandBuilder(aapt)
            .when(false).thenAdd("--dontaddthisflag", "orthisvalue").build())
        .doesNotContain("--dontaddthisflag");
  }

  @Test
  public void testThenAddFlagWithPathDoesNothingAfterWhenFalse() {
    assertThat(
        new AaptCommandBuilder(aapt)
            .when(false).thenAdd("--dontaddthisflag", manifest).build())
        .doesNotContain("--dontaddthisflag");
  }

  @Test
  public void testThenAddRepeatedDoesNothingAfterWhenFalse() {
    assertThat(
        new AaptCommandBuilder(aapt)
            .when(false).thenAddRepeated("--dontaddthisflag", ImmutableList.of("orthesevalues"))
            .build())
        .doesNotContain("--dontaddthisflag");
  }

  @Test
  public void testWhenVersionIsAtLeastAddsFlagsForEqualVersion() {
    assertThat(
        new AaptCommandBuilder(aapt).forBuildToolsVersion(new Revision(23))
            .whenVersionIsAtLeast(new Revision(23)).thenAdd("--addthisflag")
            .build())
        .contains("--addthisflag");
  }

  @Test
  public void testWhenVersionIsAtLeastAddsFlagsForGreaterVersion() {
    assertThat(
        new AaptCommandBuilder(aapt).forBuildToolsVersion(new Revision(24))
            .whenVersionIsAtLeast(new Revision(23)).thenAdd("--addthisflag")
            .build())

        .contains("--addthisflag");
  }

  @Test
  public void testWhenVersionIsAtLeastAddsFlagsForUnspecifiedVersion() {
    assertThat(
        new AaptCommandBuilder(aapt)
            .whenVersionIsAtLeast(new Revision(23)).thenAdd("--addthisflag")
            .build())
        .contains("--addthisflag");
  }

  @Test
  public void testWhenVersionIsAtLeastDoesNotAddFlagsForLesserVersion() {
    assertThat(
        new AaptCommandBuilder(aapt).forBuildToolsVersion(new Revision(22))
            .whenVersionIsAtLeast(new Revision(23)).thenAdd("--dontaddthisflag")
            .build())
        .doesNotContain("--dontaddthisflag");
  }

  @Test
  public void testWhenVariantIsAddsFlagsForEqualVariantType() {
    assertThat(
        new AaptCommandBuilder(aapt).forVariantType(VariantType.LIBRARY)
            .whenVariantIs(VariantType.LIBRARY).thenAdd("--addthisflag")
            .build())
        .contains("--addthisflag");
  }

  @Test
  public void testWhenVariantIsDoesNotAddFlagsForUnequalVariantType() {
    assertThat(
        new AaptCommandBuilder(aapt).forVariantType(VariantType.DEFAULT)
            .whenVariantIs(VariantType.LIBRARY).thenAdd("--dontaddthisflag")
            .build())
        .doesNotContain("--dontaddthisflag");
  }

  @Test
  public void testWhenVariantIsDoesNotAddFlagsForUnspecifiedVariantType() {
    assertThat(
        new AaptCommandBuilder(aapt)
            .whenVariantIs(VariantType.LIBRARY).thenAdd("--dontaddthisflag")
            .build())
        .doesNotContain("--dontaddthisflag");
  }

}
