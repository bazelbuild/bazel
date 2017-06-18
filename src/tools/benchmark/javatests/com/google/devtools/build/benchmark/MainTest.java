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

package com.google.devtools.build.benchmark;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.devtools.common.options.OptionsParsingException;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link Main}. */
@RunWith(JUnit4.class)
public class MainTest {

  private static final String TIME_FROM = "2017-02-06T18:00:00";
  private static final String TIME_TO = "2017-02-07T15:00:00";
  private static final String TIME_BETWEEN = TIME_FROM + ".." + TIME_TO;
  private static final String TIME_BETWEEN_WRONG_FORMAT = "2017-02-06T18:00..2017-02-07T15:00";

  @Test
  public void testParseArgs_MissingArgs() throws OptionsParsingException, IOException {
    try {
      Main.parseArgs(new String[] {"--workspace=workspace", "--version_between=1..2"});
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("Argument --workspace and --output should not be empty.");
    }
  }

  @Test
  public void testParseArgs_MultipleFilter() throws OptionsParsingException, IOException {
    try {
      Main.parseArgs(new String[] {
          "--output=output", "--workspace=workspace",
          "--version_between=1..2", "--time_between=" + TIME_BETWEEN});
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("Please use exact one type of version filter at a time.");
    }
  }

  @Test
  public void testParseArgs_WrongVersionBetween() throws OptionsParsingException, IOException {
    try {
      Main.parseArgs(
          new String[]{"--output=output", "--workspace=workspace", "--version_between=1.3"});
      fail("Should throw OptionsParsingException");
    } catch (OptionsParsingException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "While parsing option --version_between=1.3: "
                  + "Error parsing version_filter option: no '..' found.");
    }
  }

  @Test
  public void testParseArgs_CorrectVersionBetween() throws OptionsParsingException, IOException {
    BenchmarkOptions opt =
        Main.parseArgs(
            new String[] {"--output=output", "--workspace=workspace", "--version_between=1..3"});
    assertThat(opt.output).isEqualTo("output");
    assertThat(opt.workspace).isEqualTo("workspace");
    assertThat(opt.versionFilter.getFrom()).isEqualTo("1");
    assertThat(opt.versionFilter.getTo()).isEqualTo("3");
  }

  @Test
  public void testParseArgs_WrongTimeBetween() throws OptionsParsingException, IOException {
    try {
      Main.parseArgs(
          new String[]{"--output=output", "--workspace=workspace",
              "--time_between=" + TIME_BETWEEN_WRONG_FORMAT});
      fail("Should throw OptionsParsingException");
    } catch (OptionsParsingException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "While parsing option --time_between="
                  + TIME_BETWEEN_WRONG_FORMAT
                  + ": Error parsing datetime, format should be: yyyy-MM-ddTHH:mm:ss");
    }
  }

  @Test
  public void testParseArgs_CorrectTimeBetween() throws OptionsParsingException, IOException {
    BenchmarkOptions opt =
        Main.parseArgs(
            new String[] {"--output=output", "--workspace=workspace",
                "--time_between=" + TIME_BETWEEN});
    assertThat(opt.output).isEqualTo("output");
    assertThat(opt.workspace).isEqualTo("workspace");
    assertThat(opt.dateFilter.getFromString()).isEqualTo(TIME_FROM);
    assertThat(opt.dateFilter.getToString()).isEqualTo(TIME_TO);
  }

  @Test
  public void testParseArgs_CorrectVersions() throws OptionsParsingException, IOException {
    BenchmarkOptions opt =
        Main.parseArgs(
            new String[] {"--output=output", "--workspace=workspace",
                "--versions=v1", "--versions=v2", "--versions=v4"});
    assertThat(opt.output).isEqualTo("output");
    assertThat(opt.workspace).isEqualTo("workspace");
    assertThat(opt.versions).containsExactly("v1", "v2", "v4");
  }
}
