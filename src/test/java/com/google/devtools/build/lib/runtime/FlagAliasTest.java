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
        ImmutableList.of("c0", "--rc_source=/somewhere/.blazerc", "--flag_alias=foo=bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "--flag_alias is experimental. Set --experimental_enable_flag_alias to true to"
                    + " make use of it. Detected aliases: --flag_alias=foo=bar"));
  }

  @Test
  public void useAliasWithSetDisabledFeature() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--noexperimental_enable_flag_alias",
            "--flag_alias=foo=bar");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "--flag_alias is experimental. Set --experimental_enable_flag_alias to true to"
                    + " make use of it. Detected aliases: --flag_alias=foo=bar"));
  }

  @Test
  public void useAliasWithSetEnabledFeature() {
    ImmutableList<String> args =
        ImmutableList.of(
            "c0",
            "--rc_source=/somewhere/.blazerc",
            "--experimental_enable_flag_alias",
            "--flag_alias=foo=bar");
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
            "--flag_alias=foo=bar",
            "--flag_alias=baz=baz2");
    optionHandler.parseOptions(args, eventHandler);
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "--flag_alias is experimental. Set --experimental_enable_flag_alias to true to"
                    + " make use of it. Detected aliases: --flag_alias=foo=bar,"
                    + " --flag_alias=baz=baz2"));
  }
}
