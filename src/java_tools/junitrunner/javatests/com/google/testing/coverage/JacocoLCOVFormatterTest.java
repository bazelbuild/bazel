// Copyright 2020 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.coverage;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.io.PrintWriter;
import java.io.StringWriter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the source path handling logic in {@link JacocoLCOVFormatter}. */
@RunWith(JUnit4.class)
public class JacocoLCOVFormatterTest {

  @Test
  public void testSimpleUnixpath() throws Exception {
    CoverageData coverage =
        CoverageData.builder()
            .addMethod("Class::method", 3, true)
            .addLine(3, true)
            .addLine(4, true)
            .build();
    JacocoLCOVFormatter formatter =
        new JacocoLCOVFormatter(ImmutableSet.of("/parent/dir/com/example/Foo.java"));

    StringWriter stringWriter = new StringWriter();
    try (PrintWriter printWriter = new PrintWriter(stringWriter)) {
      formatter.writeCoverageData(printWriter, ImmutableMap.of("com/example/Foo.java", coverage));
    }

    String expected =
        """
        SF:/parent/dir/com/example/Foo.java
        FN:3,Class::method
        FNDA:1,Class::method
        DA:3,1
        DA:4,1
        end_of_record
        """;
    assertThat(stringWriter.toString()).isEqualTo(expected);
  }

  @Test
  public void testSimpleWindowsPath() throws Exception {
    CoverageData coverage =
        CoverageData.builder()
            .addMethod("Class::method", 3, true)
            .addLine(3, true)
            .addLine(4, true)
            .build();
    JacocoLCOVFormatter formatter =
        new JacocoLCOVFormatter(ImmutableSet.of("C:/parent/dir/com/example/Foo.java"));

    StringWriter stringWriter = new StringWriter();
    try (PrintWriter printWriter = new PrintWriter(stringWriter)) {
      formatter.writeCoverageData(printWriter, ImmutableMap.of("com/example/Foo.java", coverage));
    }

    String expected =
        """
        SF:C:/parent/dir/com/example/Foo.java
        FN:3,Class::method
        FNDA:1,Class::method
        DA:3,1
        DA:4,1
        end_of_record
        """;
    assertThat(stringWriter.toString()).isEqualTo(expected);
  }

  @Test
  public void testMappedUnixPath() throws Exception {
    CoverageData coverage =
        CoverageData.builder()
            .addMethod("Class::method", 3, true)
            .addLine(3, true)
            .addLine(4, true)
            .build();
    JacocoLCOVFormatter formatter =
        new JacocoLCOVFormatter(
            ImmutableSet.of("/some/other/dir/Foo.java////com/example/Foo.java"));

    StringWriter stringWriter = new StringWriter();
    try (PrintWriter printWriter = new PrintWriter(stringWriter)) {
      formatter.writeCoverageData(printWriter, ImmutableMap.of("com/example/Foo.java", coverage));
    }

    String expected =
        """
        SF:/some/other/dir/Foo.java
        FN:3,Class::method
        FNDA:1,Class::method
        DA:3,1
        DA:4,1
        end_of_record
        """;
    assertThat(stringWriter.toString()).isEqualTo(expected);
  }

  @Test
  public void testMappedWindowsPath() throws Exception {
    CoverageData coverage =
        CoverageData.builder()
            .addMethod("Class::method", 3, true)
            .addLine(3, true)
            .addLine(4, true)
            .build();
    JacocoLCOVFormatter formatter =
        new JacocoLCOVFormatter(
            ImmutableSet.of("C:/some/other/dir/Foo.java////com/example/Foo.java"));

    StringWriter stringWriter = new StringWriter();
    try (PrintWriter printWriter = new PrintWriter(stringWriter)) {
      formatter.writeCoverageData(printWriter, ImmutableMap.of("com/example/Foo.java", coverage));
    }

    String expected =
        """
        SF:C:/some/other/dir/Foo.java
        FN:3,Class::method
        FNDA:1,Class::method
        DA:3,1
        DA:4,1
        end_of_record
        """;
    assertThat(stringWriter.toString()).isEqualTo(expected);
  }

  @Test
  public void testNoMatchGivesNoOutput() throws Exception {
    CoverageData coverage =
        CoverageData.builder()
            .addMethod("Class::method", 3, true)
            .addLine(3, true)
            .addLine(4, true)
            .build();
    JacocoLCOVFormatter formatter =
        new JacocoLCOVFormatter(ImmutableSet.of("/path/does/not/match/anything.txt"));

    StringWriter stringWriter = new StringWriter();
    try (PrintWriter printWriter = new PrintWriter(stringWriter)) {
      formatter.writeCoverageData(printWriter, ImmutableMap.of("com/example/Foo.java", coverage));
    }

    assertThat(stringWriter.toString()).isEmpty();
  }

  @Test
  public void testEmptySourcePathsGivesNoOutput() throws Exception {
    CoverageData coverage =
        CoverageData.builder()
            .addMethod("Class::method", 3, true)
            .addLine(3, true)
            .addLine(4, true)
            .build();
    JacocoLCOVFormatter formatter = new JacocoLCOVFormatter(ImmutableSet.of());

    StringWriter stringWriter = new StringWriter();
    try (PrintWriter printWriter = new PrintWriter(stringWriter)) {
      formatter.writeCoverageData(printWriter, ImmutableMap.of("com/example/Foo.java", coverage));
    }

    assertThat(stringWriter.toString()).isEmpty();
  }

  @Test
  public void testNoSourcePathsOutputsOriginalName() throws Exception {
    CoverageData coverage =
        CoverageData.builder()
            .addMethod("Class::method", 3, true)
            .addLine(3, true)
            .addLine(4, true)
            .build();
    JacocoLCOVFormatter formatter = new JacocoLCOVFormatter();

    StringWriter stringWriter = new StringWriter();
    try (PrintWriter printWriter = new PrintWriter(stringWriter)) {
      formatter.writeCoverageData(printWriter, ImmutableMap.of("com/example/Foo.java", coverage));
    }

    String expected =
        """
        SF:com/example/Foo.java
        FN:3,Class::method
        FNDA:1,Class::method
        DA:3,1
        DA:4,1
        end_of_record
        """;
    assertThat(stringWriter.toString()).isEqualTo(expected);
  }

  @Test
  public void testSourcePathEqualsPackagePath() throws Exception {
    CoverageData coverage =
        CoverageData.builder()
            .addMethod("Class::method", 3, true)
            .addLine(3, true)
            .addLine(4, true)
            .build();
    JacocoLCOVFormatter formatter =
        new JacocoLCOVFormatter(ImmutableSet.of("com/example/Foo.java"));

    StringWriter stringWriter = new StringWriter();
    try (PrintWriter printWriter = new PrintWriter(stringWriter)) {
      formatter.writeCoverageData(printWriter, ImmutableMap.of("com/example/Foo.java", coverage));
    }

    String expected =
        """
        SF:com/example/Foo.java
        FN:3,Class::method
        FNDA:1,Class::method
        DA:3,1
        DA:4,1
        end_of_record
        """;
    assertThat(stringWriter.toString()).isEqualTo(expected);
  }
}
