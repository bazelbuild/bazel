// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.commands;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.bazel.commands.ModqueryCommand.InvalidArgumentException;
import com.google.devtools.build.lib.bazel.commands.ModqueryOptions.QueryType;
import com.google.devtools.build.lib.server.FailureDetails.ModqueryCommand.Code;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ModqueryCommand}. */
@RunWith(JUnit4.class)
public class ModqueryCommandTest {

  public final ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex =
      ImmutableMap.of(
          "A",
          ImmutableSet.of(ModuleKey.ROOT),
          "B",
          ImmutableSet.of(
              ModuleKey.create("B", Version.parse("1.0")),
              ModuleKey.create("B", Version.parse("2.0"))),
          "C",
          ImmutableSet.of(ModuleKey.create("C", Version.EMPTY)));

  public ModqueryCommandTest() throws ParseException {}

  @Test
  public void testAllPathsNoArgsThrowsMissingArguments() {
    InvalidArgumentException e =
        assertThrows(
            InvalidArgumentException.class,
            () ->
                ModqueryCommand.parseTargetArgs(
                    ImmutableList.of(), QueryType.ALL_PATHS.getArgNumber(), modulesIndex));
    assertThat(e.getCode()).isEqualTo(Code.MISSING_ARGUMENTS);
  }

  @Test
  public void testTreeNoArgs() throws InvalidArgumentException, OptionsParsingException {
    ModqueryCommand.parseTargetArgs(
        ImmutableList.of(), QueryType.TREE.getArgNumber(), modulesIndex);
  }

  @Test
  public void testTreeWithArgsThrowsTooManyArguments() {
    ImmutableList<String> args = ImmutableList.of("A");
    InvalidArgumentException e =
        assertThrows(
            InvalidArgumentException.class,
            () ->
                ModqueryCommand.parseTargetArgs(args, QueryType.TREE.getArgNumber(), modulesIndex));
    assertThat(e.getCode()).isEqualTo(Code.TOO_MANY_ARGUMENTS);
  }

  @Test
  public void testDepsArgWrongFormat_noVersion() {
    ImmutableList<String> args = ImmutableList.of("A@");
    assertThrows(
        OptionsParsingException.class,
        () -> ModqueryCommand.parseTargetArgs(args, QueryType.DEPS.getArgNumber(), modulesIndex));
  }

  @Test
  public void testDepsArgInvalid_missingModule() {
    ImmutableList<String> args = ImmutableList.of("D");
    InvalidArgumentException e =
        assertThrows(
            InvalidArgumentException.class,
            () ->
                ModqueryCommand.parseTargetArgs(args, QueryType.DEPS.getArgNumber(), modulesIndex));
    assertThat(e.getCode()).isEqualTo(Code.INVALID_ARGUMENTS);
  }

  @Test
  public void testDepsArgInvalid_missingModuleVersion() {
    ImmutableList<String> args = ImmutableList.of("B@3.0");
    InvalidArgumentException e =
        assertThrows(
            InvalidArgumentException.class,
            () ->
                ModqueryCommand.parseTargetArgs(args, QueryType.DEPS.getArgNumber(), modulesIndex));
    assertThat(e.getCode()).isEqualTo(Code.INVALID_ARGUMENTS);
  }

  @Test
  public void testDepsArgInvalid_invalidListFormat() {
    ImmutableList<String> args = ImmutableList.of("B@1.0;B@2.0");
    assertThrows(
        OptionsParsingException.class,
        () -> ModqueryCommand.parseTargetArgs(args, QueryType.DEPS.getArgNumber(), modulesIndex));
  }

  @Test
  public void testDepsListArg_ok() throws InvalidArgumentException, OptionsParsingException {
    ImmutableList<String> args = ImmutableList.of("A,B@1.0,B@2.0,C@_");
    ModqueryCommand.parseTargetArgs(args, QueryType.DEPS.getArgNumber(), modulesIndex);
  }
}
