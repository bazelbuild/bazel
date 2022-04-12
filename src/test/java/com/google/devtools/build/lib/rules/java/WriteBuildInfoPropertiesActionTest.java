// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.java;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Properties;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WriteBuildInfoPropertiesAction} utilities methods */
@RunWith(JUnit4.class)
public class WriteBuildInfoPropertiesActionTest extends FoundationTestCase {

  private static final Joiner LINE_JOINER = Joiner.on("\r\n");
  private static final Joiner LINEFEED_JOINER = Joiner.on("\n");

  private void assertStripFirstLine(String expected, String... testCases) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (WriteBuildInfoPropertiesAction.StripFirstLineWriter writer =
        new WriteBuildInfoPropertiesAction.StripFirstLineWriter(out)) {
      for (String testCase : testCases) {
        writer.write(testCase);
      }
    }
    assertThat(new String(out.toByteArray(), UTF_8)).isEqualTo(expected);
  }

  @Test
  public void testStripFirstLine() throws IOException {
    assertStripFirstLine("", "");
    assertStripFirstLine("", "no linefeed");
    assertStripFirstLine("", "no", "linefeed");
    assertStripFirstLine(
        LINEFEED_JOINER.join("toto", "titi"),
        LINEFEED_JOINER.join("# timestamp comment", "toto", "titi"));
    assertStripFirstLine(
        LINE_JOINER.join("toto", "titi"), LINE_JOINER.join("# timestamp comment", "toto", "titi"));
    assertStripFirstLine(
        LINEFEED_JOINER.join("toto", "titi"), "# timestamp comment\n", "toto\n", "titi");
    assertStripFirstLine(
        LINE_JOINER.join("toto", "titi"), "# timestamp comment\r\n", "toto\r\n", "titi");
  }

  @Test
  public void deterministicProperties() throws IOException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    Properties underTest = new WriteBuildInfoPropertiesAction.DeterministicProperties();
    underTest.put("second", "keyb");
    underTest.put("first", "keya");
    try (WriteBuildInfoPropertiesAction.StripFirstLineWriter writer =
        new WriteBuildInfoPropertiesAction.StripFirstLineWriter(bytes)) {
      underTest.store(writer, null);
    }
    assertThat(new String(bytes.toByteArray(), UTF_8)).isEqualTo("first=keya\nsecond=keyb\n");
  }
}
