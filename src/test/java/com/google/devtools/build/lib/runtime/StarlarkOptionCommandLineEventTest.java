// Copyright 2019 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.BazelStartupOptionsModule.Options;
import com.google.devtools.build.lib.runtime.CommandLineEvent.CanonicalCommandLineEvent;
import com.google.devtools.build.lib.runtime.CommandLineEvent.OriginalCommandLineEvent;
import com.google.devtools.build.lib.runtime.proto.CommandLineOuterClass.CommandLine;
import com.google.devtools.build.lib.runtime.proto.CommandLineOuterClass.Option;
import com.google.devtools.build.lib.starlark.util.StarlarkOptionsTestCase;
import com.google.devtools.common.options.OptionsParser;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link CommandLineEvent}'s construction of the command lines which contain
 * Starlark-style flags.
 */
@RunWith(JUnit4.class)
public class StarlarkOptionCommandLineEventTest extends StarlarkOptionsTestCase {

  @Test
  public void testStarlarkOptions_original() throws Exception {
    OptionsParser fakeStartupOptions =
        OptionsParser.builder()
            .optionsClasses(BlazeServerStartupOptions.class, Options.class)
            .build();

    writeBasicIntFlag();

    parseStarlarkOptions("--//test:my_int_setting=666");

    CommandLine line =
        new OriginalCommandLineEvent(
                "testblaze",
                fakeStartupOptions,
                "someCommandName",
                ImmutableList.of(),
                false,
                optionsParser.asListOfExplicitOptions(),
                optionsParser.getExplicitCommandLineStarlarkOptions(),
                optionsParser.getStarlarkOptionsAllowingMultiple(),
                Optional.empty())
            .asStreamProto(null)
            .getStructuredCommandLine();

    // Command options should appear in section 3. See
    // CommandLineEventTest#testOptionsAtVariousPriorities_OriginalCommandLine.
    // Verify that the starlark flag was processed as expected.
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(1);
    assertThat(line.getSections(3).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--//test:my_int_setting=666");
    assertThat(line.getSections(3).getOptionList().getOption(0).getOptionName())
        .isEqualTo("//test:my_int_setting");
    assertThat(line.getSections(3).getOptionList().getOption(0).getOptionValue()).isEqualTo("666");
  }

  /**
   * {@link OriginalCommandLineEvent} contains options explicitly set on the command line but not
   * options inherited through bazelrcs.
   */
  @Test
  public void testStarlarkOptionsFromCommandLineAndBazelRc_original() throws Exception {
    OptionsParser fakeStartupOptions =
        OptionsParser.builder()
            .optionsClasses(BlazeServerStartupOptions.class, Options.class)
            .build();
    scratch.file(
        "test/build_setting.bzl",
        """
        def _build_setting_impl(ctx):
            return []

        int_flag = rule(
            implementation = _build_setting_impl,
            build_setting = config.int(flag = True),
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:build_setting.bzl", "int_flag")

        int_flag(
            name = "cmdflag",
            build_setting_default = 10,
        )

        int_flag(
            name = "bazelrcflag",
            build_setting_default = 20,
        )
        """);

    var unused =
        parseStarlarkOptions(
            /* commandLineOptions= */ "--//test:cmdflag=666",
            /* bazelrcOptions= */ "--//test:bazelrcflag=777");
    CommandLine line =
        new OriginalCommandLineEvent(
                "testblaze",
                fakeStartupOptions,
                "someCommandName",
                ImmutableList.of(),
                false,
                optionsParser.asListOfExplicitOptions(),
                optionsParser.getExplicitCommandLineStarlarkOptions(),
                optionsParser.getStarlarkOptionsAllowingMultiple(),
                Optional.empty())
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(1);
    assertThat(line.getSections(3).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--//test:cmdflag=666");
  }

  @Test
  public void testStarlarkOptions_canonical() throws Exception {
    OptionsParser fakeStartupOptions =
        OptionsParser.builder()
            .optionsClasses(BlazeServerStartupOptions.class, Options.class)
            .build();

    writeBasicIntFlag();

    parseStarlarkOptions("--//test:my_int_setting=666");

    CommandLine line =
        new CanonicalCommandLineEvent(
                "testblaze",
                fakeStartupOptions,
                "someCommandName",
                ImmutableList.of(),
                false,
                optionsParser.getExplicitCommandLineStarlarkOptions(),
                optionsParser.getStarlarkOptions(),
                optionsParser.getStarlarkOptionsAllowingMultiple(),
                optionsParser.asListOfCanonicalOptions(),
                /* replaceable= */ false)
            .asStreamProto(null)
            .getStructuredCommandLine();
    assertThat(line.getCommandLineLabel()).isEqualTo("canonical");

    // Command options should appear in section 3. See
    // CommandLineEventTest#testOptionsAtVariousPriorities_OriginalCommandLine.
    // Verify that the starlark flag was processed as expected.
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(1);
    assertThat(line.getSections(3).getOptionList().getOption(0).getCombinedForm())
        .isEqualTo("--//test:my_int_setting=666");
    assertThat(line.getSections(3).getOptionList().getOption(0).getOptionName())
        .isEqualTo("//test:my_int_setting");
    assertThat(line.getSections(3).getOptionList().getOption(0).getOptionValue()).isEqualTo("666");
  }

  @Test
  public void testStarlarkOptions_canonical_defaultValue() throws Exception {
    OptionsParser fakeStartupOptions =
        OptionsParser.builder()
            .optionsClasses(BlazeServerStartupOptions.class, Options.class)
            .build();

    writeBasicIntFlag();

    parseStarlarkOptions("--//test:my_int_setting=42");

    CommandLine line =
        new CanonicalCommandLineEvent(
                "testblaze",
                fakeStartupOptions,
                "someCommandName",
                ImmutableList.of(),
                false,
                optionsParser.getExplicitCommandLineStarlarkOptions(),
                optionsParser.getStarlarkOptions(),
                optionsParser.getStarlarkOptionsAllowingMultiple(),
                optionsParser.asListOfCanonicalOptions(),
                /* replaceable= */ false)
            .asStreamProto(null)
            .getStructuredCommandLine();
    assertThat(line.getCommandLineLabel()).isEqualTo("canonical");

    // Command options should appear in section 3. See
    // CommandLineEventTest#testOptionsAtVariousPriorities_OriginalCommandLine.
    // Verify that the starlark flag was processed as expected.
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(0);
  }

  /**
   * {@link CanonicalCommandLineEvent} includes both options set explicitly on the command line and
   * options inherited through bazelrcs.
   */
  @Test
  public void testStarlarkOptionsFromCommandLineAndBazelRc_canonical() throws Exception {
    OptionsParser fakeStartupOptions =
        OptionsParser.builder()
            .optionsClasses(BlazeServerStartupOptions.class, Options.class)
            .build();
    scratch.file(
        "test/build_setting.bzl",
        """
        def _build_setting_impl(ctx):
            return []

        int_flag = rule(
            implementation = _build_setting_impl,
            build_setting = config.int(flag = True),
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:build_setting.bzl", "int_flag")

        int_flag(
            name = "cmdflag",
            build_setting_default = 10,
        )

        int_flag(
            name = "bazelrcflag",
            build_setting_default = 20,
        )
        """);

    var unused =
        parseStarlarkOptions(
            /* commandLineOptions= */ "--//test:cmdflag=666",
            /* bazelrcOptions= */ "--//test:bazelrcflag=777");
    CommandLine line =
        new CanonicalCommandLineEvent(
                "testblaze",
                fakeStartupOptions,
                "someCommandName",
                ImmutableList.of(),
                false,
                optionsParser.getExplicitCommandLineStarlarkOptions(),
                optionsParser.getStarlarkOptions(),
                optionsParser.getStarlarkOptionsAllowingMultiple(),
                optionsParser.asListOfCanonicalOptions(),
                /* replaceable= */ false)
            .asStreamProto(null)
            .getStructuredCommandLine();

    assertThat(
            line.getSections(3).getOptionList().getOptionList().stream()
                .map(o -> o.getCombinedForm()))
        .containsExactly("--//test:cmdflag=666", "--//test:bazelrcflag=777");
  }

  @Test
  public void testStarlarkOptions_multipleAndRepeatable() throws Exception {
    OptionsParser fakeStartupOptions =
        OptionsParser.builder().optionsClasses(BlazeServerStartupOptions.class).build();
    scratch.file(
        "flags/build_settings.bzl",
        """
        def _impl(ctx):
            return []

        allow_multiple_flag = rule(
            implementation = _impl,
            build_setting = config.string(flag = True, allow_multiple = True),
        )

        repeatable_flag = rule(
            implementation = _impl,
            build_setting = config.string_list(flag = True, repeatable = True),
        )

        singular_flag = rule(
            implementation = _impl,
            build_setting = config.string(flag = True),
        )
        """);

    scratch.file(
        "flags/BUILD",
        """
        load(":build_settings.bzl", "allow_multiple_flag", "repeatable_flag", "singular_flag")
        allow_multiple_flag(name = "allow_multiple", build_setting_default = "")
        repeatable_flag(name = "repeatable", build_setting_default = [])
        singular_flag(name = "singular", build_setting_default = "")
        """);

    var unused =
        parseStarlarkOptions(
            "--//flags:singular=abc --//flags:allow_multiple=foo --//flags:allow_multiple=bar"
                + " --//flags:repeatable=good --//flags:repeatable=bye");

    CommandLine line =
        new OriginalCommandLineEvent(
                "testblaze",
                fakeStartupOptions,
                "someCommandName",
                ImmutableList.of(),
                false,
                optionsParser.asListOfExplicitOptions(),
                optionsParser.getStarlarkOptions(),
                optionsParser.getStarlarkOptionsAllowingMultiple(),
                Optional.empty())
            .asStreamProto(null)
            .getStructuredCommandLine();
    assertThat(
            line.getSections(3).getOptionList().getOptionList().stream()
                .map(Option::getCombinedForm)
                .collect(toImmutableList()))
        .containsExactly(
            // Flags names are sorted; order of multiple values is preserved.
            "--//flags:allow_multiple=foo",
            "--//flags:allow_multiple=bar",
            "--//flags:repeatable=good",
            "--//flags:repeatable=bye",
            "--//flags:singular=abc")
        .inOrder();
  }
}
