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

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.bazel.BazelStartupOptionsModule.Options;
import com.google.devtools.build.lib.runtime.CommandLineEvent.CanonicalCommandLineEvent;
import com.google.devtools.build.lib.runtime.CommandLineEvent.OriginalCommandLineEvent;
import com.google.devtools.build.lib.runtime.proto.CommandLineOuterClass.CommandLine;
import com.google.devtools.build.lib.skylark.util.StarlarkOptionsTestCase;
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
                "testblaze", fakeStartupOptions, "someCommandName", optionsParser, Optional.empty())
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
                "testblaze", fakeStartupOptions, "someCommandName", optionsParser)
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
                "testblaze", fakeStartupOptions, "someCommandName", optionsParser)
            .asStreamProto(null)
            .getStructuredCommandLine();
    assertThat(line.getCommandLineLabel()).isEqualTo("canonical");

    // Command options should appear in section 3. See
    // CommandLineEventTest#testOptionsAtVariousPriorities_OriginalCommandLine.
    // Verify that the starlark flag was processed as expected.
    assertThat(line.getSections(3).getOptionList().getOptionCount()).isEqualTo(0);
  }
}
