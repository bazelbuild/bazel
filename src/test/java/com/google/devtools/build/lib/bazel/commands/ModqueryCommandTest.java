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

import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryOptions.QueryType;
import com.google.devtools.build.lib.bazel.commands.ModqueryCommand.InvalidArgumentException;
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
          "aaa",
          ImmutableSet.of(ModuleKey.ROOT),
          "bbb",
          ImmutableSet.of(
              ModuleKey.create("bbb", Version.parse("1.0")),
              ModuleKey.create("bbb", Version.parse("1.5")),
              ModuleKey.create("bbb", Version.parse("2.0"))),
          "ccc",
          ImmutableSet.of(ModuleKey.create("ccc", Version.EMPTY)));
  public final ImmutableBiMap<String, ModuleKey> rootDeps =
      ImmutableBiMap.of(
          "bbb1",
          ModuleKey.create("bbb", Version.parse("1.5")),
          "bbb2",
          ModuleKey.create("bbb", Version.parse("2.0")));
  public final ImmutableBiMap<String, ModuleKey> rootUnusedDeps =
      ImmutableBiMap.of("bbb1", ModuleKey.create("bbb", Version.parse("1.0")));

  public ModqueryCommandTest() throws ParseException {}

  @Test
  public void testAllPathsNoArgsThrowsMissingArguments() {
    InvalidArgumentException e =
        assertThrows(
            InvalidArgumentException.class,
            () ->
                ModqueryCommand.parseTargetArgs(
                    QueryType.ALL_PATHS.getArgNumber(),
                    modulesIndex,
                    ImmutableList.of(),
                    rootDeps,
                    rootUnusedDeps,
                    false));
    assertThat(e.getCode()).isEqualTo(Code.MISSING_ARGUMENTS);
  }

  @Test
  public void testTreeNoArgs() throws InvalidArgumentException, OptionsParsingException {
    var unused =
        ModqueryCommand.parseTargetArgs(
            QueryType.TREE.getArgNumber(),
            modulesIndex,
            ImmutableList.of(),
            rootDeps,
            rootUnusedDeps,
            false);
  }

  @Test
  public void testTreeWithArgsThrowsTooManyArguments() {
    ImmutableList<String> args = ImmutableList.of("aaa");
    InvalidArgumentException e =
        assertThrows(
            InvalidArgumentException.class,
            () ->
                ModqueryCommand.parseTargetArgs(
                    QueryType.TREE.getArgNumber(),
                    modulesIndex,
                    args,
                    rootDeps,
                    rootUnusedDeps,
                    false));
    assertThat(e.getCode()).isEqualTo(Code.TOO_MANY_ARGUMENTS);
  }

  @Test
  public void testDepsArgWrongFormat_noVersion() {
    ImmutableList<String> args = ImmutableList.of("aaa@");
    assertThrows(
        OptionsParsingException.class,
        () ->
            ModqueryCommand.parseTargetArgs(
                QueryType.DEPS.getArgNumber(),
                modulesIndex,
                args,
                rootDeps,
                rootUnusedDeps,
                false));
  }

  @Test
  public void testDepsArgInvalid_missingModule() {
    ImmutableList<String> args = ImmutableList.of("ddd");
    InvalidArgumentException e =
        assertThrows(
            InvalidArgumentException.class,
            () ->
                ModqueryCommand.parseTargetArgs(
                    QueryType.DEPS.getArgNumber(),
                    modulesIndex,
                    args,
                    rootDeps,
                    rootUnusedDeps,
                    false));
    assertThat(e.getCode()).isEqualTo(Code.INVALID_ARGUMENTS);
  }

  @Test
  public void testDepsArgInvalid_missingModuleVersion() {
    ImmutableList<String> args = ImmutableList.of("bbb@3.0");
    InvalidArgumentException e =
        assertThrows(
            InvalidArgumentException.class,
            () ->
                ModqueryCommand.parseTargetArgs(
                    QueryType.DEPS.getArgNumber(),
                    modulesIndex,
                    args,
                    rootDeps,
                    rootUnusedDeps,
                    false));
    assertThat(e.getCode()).isEqualTo(Code.INVALID_ARGUMENTS);
  }

  @Test
  public void testDepsArgInvalid_invalidListFormat() {
    ImmutableList<String> args = ImmutableList.of("bbb@1.0;bbb@2.0");
    assertThrows(
        OptionsParsingException.class,
        () ->
            ModqueryCommand.parseTargetArgs(
                QueryType.DEPS.getArgNumber(),
                modulesIndex,
                args,
                rootDeps,
                rootUnusedDeps,
                false));
  }

  @Test
  public void testDepsListArg_ok() throws InvalidArgumentException, OptionsParsingException {
    ImmutableList<String> args = ImmutableList.of("aaa,bbb@1.0,bbb@2.0,ccc@_");
    var unused =
        ModqueryCommand.parseTargetArgs(
            QueryType.DEPS.getArgNumber(), modulesIndex, args, rootDeps, rootUnusedDeps, false);
  }

  @Test
  public void testRepoNameArg_ok()
      throws InvalidArgumentException, OptionsParsingException, ParseException {
    ImmutableList<String> args = ImmutableList.of("bbb1");
    ImmutableSet<ModuleKey> result =
        ModqueryCommand.parseTargetArgs(
                QueryType.DEPS.getArgNumber(), modulesIndex, args, rootDeps, rootUnusedDeps, false)
            .get(0);
    assertThat(result).containsExactly(ModuleKey.create("bbb", Version.parse("1.5")));

    ImmutableSet<ModuleKey> resultUnused =
        ModqueryCommand.parseTargetArgs(
                QueryType.DEPS.getArgNumber(), modulesIndex, args, rootDeps, rootUnusedDeps, true)
            .get(0);
    assertThat(resultUnused)
        .containsExactly(
            ModuleKey.create("bbb", Version.parse("1.0")),
            ModuleKey.create("bbb", Version.parse("1.5")));
  }
}
