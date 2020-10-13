// Copyright 2020 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.events.Event;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests --flag_alias functionality in {@link BlazeOptionHandler}. */
@RunWith(JUnit4.class)
public final class FlagAliasTest extends AbstractBlazeOptionHandlerTest {

  @Test
  public void useAliasWithoutSettingFeature() {
    ImmutableList<String> args =
        ImmutableList.of("c0", "--rc_source=/somewhere/.blazerc", "--flag_alias=foo=//bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "--flag_alias is experimental. Set --experimental_enable_flag_alias to true to"
                    + " make use of it. Detected aliases: --flag_alias=foo=//bar"));
  }

  @Test
  public void useAliasWithSetDisabledFeature() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--noexperimental_enable_flag_alias",
            "--flag_alias=foo=//bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "--flag_alias is experimental. Set --experimental_enable_flag_alias to true to"
                    + " make use of it. Detected aliases: --flag_alias=foo=//bar"));
  }

  @Test
  public void useAliasWithSetDisabledFeatureRcFile() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--noexperimental_enable_flag_alias",
            "--rc_source=/somewhere/.blazerc",
            "--flag_alias=foo=//bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "--flag_alias is experimental. Set --experimental_enable_flag_alias to true to"
                    + " make use of it. Detected aliases: --flag_alias=foo=//bar"));
  }

  @Test
  public void useAliasWithSetEnabledFeature() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--experimental_enable_flag_alias",
            "--flag_alias=foo=//bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.hasErrors()).isFalse();
  }

  @Test
  public void multipleAliasesLoggedInError() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--noexperimental_enable_flag_alias",
            "--flag_alias=foo=//bar",
            "--flag_alias=baz=//baz2");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "--flag_alias is experimental. Set --experimental_enable_flag_alias to true to"
                    + " make use of it. Detected aliases: --flag_alias=foo=//bar,"
                    + " --flag_alias=baz=//baz2"));
  }

  @Test
  public void useAliasWithNonStarlarkFlag() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--experimental_enable_flag_alias",
            "--flag_alias=foo=bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "While parsing option --flag_alias=foo=bar: --flag_alias only supports Starlark"
                    + " build settings."));
  }

  @Test
  public void useAliasWithValueAssignment() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--experimental_enable_flag_alias",
            "--flag_alias=foo=//bar=7");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "While parsing option --flag_alias=foo=//bar=7: --flag_alias does not support flag"
                    + " value assignment."));
  }

  @Test
  public void useAliasWithInvalidName() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--experimental_enable_flag_alias",
            "--flag_alias=bad$foo=//bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "While parsing option --flag_alias=bad$foo=//bar: bad$foo should only consist of"
                    + " word characters to be a valid alias name."));
  }

  @Test
  public void useAliasWithoutEqualsInValue() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--experimental_enable_flag_alias",
            "--flag_alias=foo");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "While parsing option --flag_alias=foo: Variable definitions must be in"
                    + " the form of a 'name=value' assignment"));
  }

  @Test
  public void useAliasWithoutEqualsInArg() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--experimental_enable_flag_alias",
            "--flag_alias",
            "foo=//bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.hasErrors()).isFalse();
  }

  @Test
  public void useAliasWithBooleanSyntax() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--experimental_enable_flag_alias",
            "--flag_alias=foo=//bar",
            "--foo");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(parser.getResidue()).contains("--//bar");
  }

  @Test
  public void lastRepeatMappingTakesPrecedence() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--experimental_enable_flag_alias",
            "--flag_alias=foo=//bar",
            "--foo",
            "--flag_alias=foo=//baz",
            "--foo");
    ImmutableList<String> expectedResidue = ImmutableList.of("--//bar", "--//baz");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(parser.getResidue()).isEqualTo(expectedResidue);
  }

  @Test
  public void setAliasInRcFile_useInRcFile() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--experimental_enable_flag_alias",
            "--default_override=0:c0=--flag_alias=foo=//bar",
            "--default_override=0:c0=--foo",
            "--rc_source=/somewhere/.blazerc");
    ImmutableList<String> expectedResidue = ImmutableList.of("--//bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(parser.getResidue()).isEqualTo(expectedResidue);
  }

  @Test
  public void setAliasInRcFile_useOnCommandLine() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--experimental_enable_flag_alias",
            "--default_override=0:c0=--flag_alias=foo=//bar",
            "--rc_source=/somewhere/.blazerc",
            "--foo");
    ImmutableList<String> expectedResidue = ImmutableList.of("--//bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(parser.getResidue()).isEqualTo(expectedResidue);
  }

  @Test
  public void setAliasOnCommandLine_useOnCommandLine() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--experimental_enable_flag_alias",
            "--flag_alias=foo=//bar",
            "--foo=7");
    ImmutableList<String> expectedResidue = ImmutableList.of("--//bar=7");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(parser.getResidue()).isEqualTo(expectedResidue);
  }

  @Test
  public void setAliasOnCommandLine_useInRcFile() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--foo=7",
            "--rc_source=/somewhere/.blazerc",
            "--experimental_enable_flag_alias",
            "--flag_alias=foo=//bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error("--foo=7 :: Unrecognized option: --foo=7")
                .withTag(BlazeOptionHandler.BAD_OPTION_TAG));
  }

  @Test
  public void useAliasBeforeSettingOnCommandLine() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--experimental_enable_flag_alias",
            "--foo=7",
            "--flag_alias=foo=//bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error("--foo=7 :: Unrecognized option: --foo=7")
                .withTag(BlazeOptionHandler.BAD_OPTION_TAG));
  }
}
