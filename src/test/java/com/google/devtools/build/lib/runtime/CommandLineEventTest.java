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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.CommandLineEvent.CanonicalCommandLineEvent;
import com.google.devtools.build.lib.runtime.CommandLineEvent.OriginalCommandLineEvent;
import com.google.devtools.build.lib.runtime.proto.CommandLineOuterClass.CommandLine;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.TestOptions;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CommandLineEvent}'s construction of the command lines. */
@RunWith(JUnit4.class)
public class CommandLineEventTest {

  private void checkCommandLineSectionLabels(CommandLine line) {
    assertThat(line.getSectionsCount()).isEqualTo(5);

    assertThat(line.getSections(0).getSectionLabel()).isEqualTo("executable");
    assertThat(line.getSections(1).getSectionLabel()).isEqualTo("startup options");
    assertThat(line.getSections(2).getSectionLabel()).isEqualTo("command");
    assertThat(line.getSections(3).getSectionLabel()).isEqualTo("command options");
    assertThat(line.getSections(4).getSectionLabel()).isEqualTo("residual");
  }

  @Test
  public void testMostlyEmpty_OriginalCommandLine() {
    OptionsParser fakeStartupOptions =
        OptionsParser.newOptionsParser(BlazeServerStartupOptions.class);
    OptionsParser fakeCommandOptions = OptionsParser.newOptionsParser(TestOptions.class);

    CommandLine line =
        new OriginalCommandLineEvent(
                "testblaze",
                fakeStartupOptions,
                "someCommandName",
                fakeCommandOptions,
                Optional.of(ImmutableList.of()))
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(line).isNotNull();
    assertThat(line.getCommandLineLabel()).isEqualTo("original");
    checkCommandLineSectionLabels(line);
    assertThat(line.getSections(0).getChunkList().getChunk(0)).isEqualTo("testblaze");
    assertThat(line.getSections(1).getOptionList().getOptionCount()).isEqualTo(0);
    assertThat(line.getSections(2).getChunkList().getChunk(0)).isEqualTo("someCommandName");
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(0);
    assertThat(line.getSections(4).getChunkList().getChunkCount()).isEqualTo(0);
  }

  @Test
  public void testMostlyEmpty_CanonicalCommandLine() {
    OptionsParser fakeStartupOptions =
        OptionsParser.newOptionsParser(BlazeServerStartupOptions.class);
    OptionsParser fakeCommandOptions = OptionsParser.newOptionsParser(TestOptions.class);

    CommandLine line =
        new CanonicalCommandLineEvent(
                "testblaze", fakeStartupOptions, "someCommandName", fakeCommandOptions)
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(line).isNotNull();
    assertThat(line.getCommandLineLabel()).isEqualTo("canonical");
    checkCommandLineSectionLabels(line);

    assertThat(line.getSections(0).getChunkList().getChunk(0)).isEqualTo("testblaze");
    assertThat(line.getSections(1).getOptionList().getOptionCount()).isEqualTo(2);
    assertThat(line.getSections(1).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--nomaster_blazerc");
    assertThat(line.getSections(1).getOptionList().getOption(1).getCombinedForm())
        .isEqualTo("--blazerc=/dev/null");
    assertThat(line.getSections(2).getChunkList().getChunk(0)).isEqualTo("someCommandName");
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(0);
    assertThat(line.getSections(4).getChunkList().getChunkCount()).isEqualTo(0);
  }

  @Test
  public void testActiveBlazercs_OriginalCommandLine() throws OptionsParsingException {
    OptionsParser fakeStartupOptions =
        OptionsParser.newOptionsParser(BlazeServerStartupOptions.class);
    fakeStartupOptions.parse(
        "--blazerc=/some/path", "--master_blazerc", "--blazerc", "/some/other/path");
    OptionsParser fakeCommandOptions = OptionsParser.newOptionsParser(TestOptions.class);

    CommandLine line =
        new OriginalCommandLineEvent(
                "testblaze",
                fakeStartupOptions,
                "someCommandName",
                fakeCommandOptions,
                Optional.empty())
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(line).isNotNull();
    assertThat(line.getCommandLineLabel()).isEqualTo("original");
    checkCommandLineSectionLabels(line);

    // Expect the provided rc-related startup options are correctly listed
    assertThat(line.getSections(0).getChunkList().getChunk(0)).isEqualTo("testblaze");
    assertThat(line.getSections(1).getOptionList().getOptionCount()).isEqualTo(3);
    assertThat(line.getSections(1).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--blazerc=/some/path");
    assertThat(line.getSections(1).getOptionList().getOption(1).getCombinedForm())
        .isEqualTo("--master_blazerc");
    assertThat(line.getSections(1).getOptionList().getOption(2).getCombinedForm())
        .isEqualTo("--blazerc /some/other/path");
    assertThat(line.getSections(2).getChunkList().getChunk(0)).isEqualTo("someCommandName");
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(0);
    assertThat(line.getSections(4).getChunkList().getChunkCount()).isEqualTo(0);
  }

  @Test
  public void testPassedInBlazercs_OriginalCommandLine() throws OptionsParsingException {
    OptionsParser fakeStartupOptions =
        OptionsParser.newOptionsParser(BlazeServerStartupOptions.class);
    OptionsParser fakeCommandOptions = OptionsParser.newOptionsParser(TestOptions.class);

    CommandLine line =
        new OriginalCommandLineEvent(
                "testblaze",
                fakeStartupOptions,
                "someCommandName",
                fakeCommandOptions,
                Optional.of(
                    ImmutableList.of(
                        Pair.of("", "--blazerc=/some/path"),
                        Pair.of("", "--master_blazerc"),
                        Pair.of("", "--blazerc=/some/other/path"),
                        Pair.of("", "--invocation_policy=notARealPolicy"))))
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(line).isNotNull();
    assertThat(line.getCommandLineLabel()).isEqualTo("original");
    checkCommandLineSectionLabels(line);

    // Expect the provided rc-related startup options are correctly listed
    assertThat(line.getSections(0).getChunkList().getChunk(0)).isEqualTo("testblaze");
    assertThat(line.getSections(1).getOptionList().getOptionCount()).isEqualTo(4);
    assertThat(line.getSections(1).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--blazerc=/some/path");
    assertThat(line.getSections(1).getOptionList().getOption(1).getCombinedForm())
        .isEqualTo("--master_blazerc");
    assertThat(line.getSections(1).getOptionList().getOption(2).getCombinedForm())
        .isEqualTo("--blazerc=/some/other/path");
    assertThat(line.getSections(1).getOptionList().getOption(3).getCombinedForm())
        .isEqualTo("--invocation_policy=notARealPolicy");
    assertThat(line.getSections(2).getChunkList().getChunk(0)).isEqualTo("someCommandName");
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(0);
    assertThat(line.getSections(4).getChunkList().getChunkCount()).isEqualTo(0);
  }

  @Test
  public void testBlazercs_CanonicalCommandLine() throws OptionsParsingException {
    OptionsParser fakeStartupOptions =
        OptionsParser.newOptionsParser(BlazeServerStartupOptions.class);
    fakeStartupOptions.parse(
        "--blazerc=/some/path", "--master_blazerc", "--blazerc", "/some/other/path");
    OptionsParser fakeCommandOptions = OptionsParser.newOptionsParser(TestOptions.class);

    CommandLine line =
        new CanonicalCommandLineEvent(
                "testblaze", fakeStartupOptions, "someCommandName", fakeCommandOptions)
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(line).isNotNull();
    assertThat(line.getCommandLineLabel()).isEqualTo("canonical");
    checkCommandLineSectionLabels(line);

    // Expect the provided rc-related startup options are removed and replaced with the
    // rc-prevention options.
    assertThat(line.getSections(0).getChunkList().getChunk(0)).isEqualTo("testblaze");
    assertThat(line.getSections(1).getOptionList().getOptionCount()).isEqualTo(2);
    assertThat(line.getSections(1).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--nomaster_blazerc");
    assertThat(line.getSections(1).getOptionList().getOption(1).getCombinedForm())
        .isEqualTo("--blazerc=/dev/null");
    assertThat(line.getSections(2).getChunkList().getChunk(0)).isEqualTo("someCommandName");
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(0);
    assertThat(line.getSections(4).getChunkList().getChunkCount()).isEqualTo(0);
  }

  @Test
  public void testOptionsAtVariousPriorities_OriginalCommandLine() throws OptionsParsingException {
    OptionsParser fakeStartupOptions =
        OptionsParser.newOptionsParser(BlazeServerStartupOptions.class);
    OptionsParser fakeCommandOptions = OptionsParser.newOptionsParser(TestOptions.class);
    fakeCommandOptions.parse(
        OptionPriority.COMMAND_LINE,
        "command line",
        ImmutableList.of("--test_string=foo", "--test_multiple_string=bar"));
    fakeCommandOptions.parse(
        OptionPriority.INVOCATION_POLICY,
        "fake invocation policy",
        ImmutableList.of("--expanded_c=2"));
    fakeCommandOptions.parse(
        OptionPriority.RC_FILE, "fake rc file", ImmutableList.of("--test_multiple_string=baz"));

    CommandLine line =
        new OriginalCommandLineEvent(
                "testblaze",
                fakeStartupOptions,
                "someCommandName",
                fakeCommandOptions,
                Optional.of(ImmutableList.of()))
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(line).isNotNull();
    assertThat(line.getCommandLineLabel()).isEqualTo("original");
    checkCommandLineSectionLabels(line);

    assertThat(line.getSections(0).getChunkList().getChunk(0)).isEqualTo("testblaze");
    assertThat(line.getSections(1).getOptionList().getOptionCount()).isEqualTo(0);
    assertThat(line.getSections(2).getChunkList().getChunk(0)).isEqualTo("someCommandName");
    // Expect the rc file options and invocation policy options to not be listed with the explicit
    // command line options.
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(2);
    assertThat(line.getSections(3).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--test_string=foo");
    assertThat(line.getSections(3).getOptionList().getOption(1).getCombinedForm())
        .isEqualTo("--test_multiple_string=bar");
    assertThat(line.getSections(4).getChunkList().getChunkCount()).isEqualTo(0);
  }

  @Test
  public void testOptionsAtVariousPriorities_CanonicalCommandLine() throws OptionsParsingException {
    OptionsParser fakeStartupOptions =
        OptionsParser.newOptionsParser(BlazeServerStartupOptions.class);
    OptionsParser fakeCommandOptions = OptionsParser.newOptionsParser(TestOptions.class);
    fakeCommandOptions.parse(
        OptionPriority.COMMAND_LINE,
        "command line",
        ImmutableList.of("--test_string=foo", "--test_multiple_string=bar"));
    fakeCommandOptions.parse(
        OptionPriority.INVOCATION_POLICY,
        "fake invocation policy",
        ImmutableList.of("--expanded_c=2"));
    fakeCommandOptions.parse(
        OptionPriority.RC_FILE, "fake rc file", ImmutableList.of("--test_multiple_string=baz"));

    CommandLine line =
        new CanonicalCommandLineEvent(
                "testblaze", fakeStartupOptions, "someCommandName", fakeCommandOptions)
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(line).isNotNull();
    assertThat(line.getCommandLineLabel()).isEqualTo("canonical");
    checkCommandLineSectionLabels(line);

    assertThat(line.getSections(0).getChunkList().getChunk(0)).isEqualTo("testblaze");
    assertThat(line.getSections(1).getOptionList().getOptionCount()).isEqualTo(2);
    assertThat(line.getSections(2).getChunkList().getChunk(0)).isEqualTo("someCommandName");
    // In the canonical line, expect the rc option to show up before the higher priority options.
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(4);
    assertThat(line.getSections(3).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--test_multiple_string=baz");
    assertThat(line.getSections(3).getOptionList().getOption(1).getCombinedForm())
        .isEqualTo("--test_string=foo");
    assertThat(line.getSections(3).getOptionList().getOption(2).getCombinedForm())
        .isEqualTo("--test_multiple_string=bar");
    assertThat(line.getSections(3).getOptionList().getOption(3).getCombinedForm())
        .isEqualTo("--expanded_c=2");

    assertThat(line.getSections(4).getChunkList().getChunkCount()).isEqualTo(0);
  }

  @Test
  public void testExpansionOption_OriginalCommandLine() throws OptionsParsingException {
    OptionsParser fakeStartupOptions =
        OptionsParser.newOptionsParser(BlazeServerStartupOptions.class);
    OptionsParser fakeCommandOptions = OptionsParser.newOptionsParser(TestOptions.class);
    fakeCommandOptions.parse(
        OptionPriority.COMMAND_LINE, "command line", ImmutableList.of("--test_expansion"));

    CommandLine line =
        new OriginalCommandLineEvent(
                "testblaze",
                fakeStartupOptions,
                "someCommandName",
                fakeCommandOptions,
                Optional.of(ImmutableList.of()))
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(line).isNotNull();
    assertThat(line.getCommandLineLabel()).isEqualTo("original");
    checkCommandLineSectionLabels(line);

    assertThat(line.getSections(0).getChunkList().getChunk(0)).isEqualTo("testblaze");
    assertThat(line.getSections(1).getOptionList().getOptionCount()).isEqualTo(0);
    assertThat(line.getSections(2).getChunkList().getChunk(0)).isEqualTo("someCommandName");
    // Expect the rc file option to not be listed with the explicit command line options.
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(1);
    assertThat(line.getSections(3).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--test_expansion");
    assertThat(line.getSections(4).getChunkList().getChunkCount()).isEqualTo(0);
  }

  @Test
  public void testExpansionOption_CanonicalCommandLine() throws OptionsParsingException {
    OptionsParser fakeStartupOptions =
        OptionsParser.newOptionsParser(BlazeServerStartupOptions.class);
    OptionsParser fakeCommandOptions = OptionsParser.newOptionsParser(TestOptions.class);
    fakeCommandOptions.parse(
        OptionPriority.COMMAND_LINE, "command line", ImmutableList.of("--test_expansion"));

    CommandLine line =
        new CanonicalCommandLineEvent(
                "testblaze", fakeStartupOptions, "someCommandName", fakeCommandOptions)
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(line).isNotNull();
    assertThat(line.getCommandLineLabel()).isEqualTo("canonical");
    checkCommandLineSectionLabels(line);

    assertThat(line.getSections(0).getChunkList().getChunk(0)).isEqualTo("testblaze");
    assertThat(line.getSections(1).getOptionList().getOptionCount()).isEqualTo(2);
    assertThat(line.getSections(2).getChunkList().getChunk(0)).isEqualTo("someCommandName");

    // TODO(b/19881919) Expansion options shouldn't be listed along with their expansions, this
    // could cause duplicate values for repeatable flags. There should be 4 flags here, without
    // test_expansion listed.
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(5);
    assertThat(line.getSections(3).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--test_expansion");
    assertThat(line.getSections(3).getOptionList().getOption(1).getCombinedForm())
        .isEqualTo("--noexpanded_a");
    assertThat(line.getSections(3).getOptionList().getOption(2).getCombinedForm())
        .isEqualTo("--expanded_b=false");
    assertThat(line.getSections(3).getOptionList().getOption(3).getCombinedForm())
        .isEqualTo("--expanded_c 42");
    assertThat(line.getSections(3).getOptionList().getOption(4).getCombinedForm())
        .isEqualTo("--expanded_d bar");
    assertThat(line.getSections(4).getChunkList().getChunkCount()).isEqualTo(0);
  }

  @Test
  public void testOptionWithImplicitRequirement_OriginalCommandLine()
      throws OptionsParsingException {
    OptionsParser fakeStartupOptions =
        OptionsParser.newOptionsParser(BlazeServerStartupOptions.class);
    OptionsParser fakeCommandOptions = OptionsParser.newOptionsParser(TestOptions.class);
    fakeCommandOptions.parse(
        OptionPriority.COMMAND_LINE,
        "command line",
        ImmutableList.of("--test_implicit_requirement=foo"));

    CommandLine line =
        new OriginalCommandLineEvent(
                "testblaze",
                fakeStartupOptions,
                "someCommandName",
                fakeCommandOptions,
                Optional.of(ImmutableList.of()))
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(line).isNotNull();
    assertThat(line.getCommandLineLabel()).isEqualTo("original");
    checkCommandLineSectionLabels(line);

    assertThat(line.getSections(0).getChunkList().getChunk(0)).isEqualTo("testblaze");
    assertThat(line.getSections(1).getOptionList().getOptionCount()).isEqualTo(0);
    assertThat(line.getSections(2).getChunkList().getChunk(0)).isEqualTo("someCommandName");
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(1);
    assertThat(line.getSections(3).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--test_implicit_requirement=foo");
    assertThat(line.getSections(4).getChunkList().getChunkCount()).isEqualTo(0);
  }

  @Test
  public void testOptionWithImplicitRequirement_CanonicalCommandLine()
      throws OptionsParsingException {
    OptionsParser fakeStartupOptions =
        OptionsParser.newOptionsParser(BlazeServerStartupOptions.class);
    OptionsParser fakeCommandOptions = OptionsParser.newOptionsParser(TestOptions.class);
    fakeCommandOptions.parse(
        OptionPriority.COMMAND_LINE,
        "command line",
        ImmutableList.of("--test_implicit_requirement=foo"));

    CommandLine line =
        new CanonicalCommandLineEvent(
                "testblaze", fakeStartupOptions, "someCommandName", fakeCommandOptions)
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(line).isNotNull();
    assertThat(line.getCommandLineLabel()).isEqualTo("canonical");
    checkCommandLineSectionLabels(line);

    // Unlike expansion flags, implicit requirements are not listed separately.
    assertThat(line.getSections(0).getChunkList().getChunk(0)).isEqualTo("testblaze");
    assertThat(line.getSections(1).getOptionList().getOptionCount()).isEqualTo(2);
    assertThat(line.getSections(2).getChunkList().getChunk(0)).isEqualTo("someCommandName");
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(1);
    assertThat(line.getSections(3).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--test_implicit_requirement=foo");
    assertThat(line.getSections(4).getChunkList().getChunkCount()).isEqualTo(0);
  }
}
