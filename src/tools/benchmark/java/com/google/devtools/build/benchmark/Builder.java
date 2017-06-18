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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.shell.CommandException;
import java.io.IOException;
import java.nio.file.Path;

/** Interface that includes methods for a building tool. */
interface Builder {

  /** Prepare anything the build needs. */
  void prepare() throws IOException, CommandException;

  /** Returns the binary path of the build tool of a specific {@code codeVersion}. */
  Path getBuildBinary(String codeVersion) throws IOException, CommandException;

  /** Returns the code versions of the build tool between versions {@code (from, to]}. */
  ImmutableList<String> getCodeVersionsBetweenVersions(VersionFilter versionFilter)
      throws CommandException;

  /** Returns the code versions of the build tool between dates {@code [from, to]}. */
  ImmutableList<String> getCodeVersionsBetweenDates(DateFilter dateFilter)
      throws CommandException;

  /** Return the datetime of all {@code codeVersions} */
  ImmutableList<String> getDatetimeForCodeVersions(ImmutableList<String> codeVersions)
      throws CommandException;

  /** Returns a command for build under specific config. */
  ImmutableList<String> getCommandFromConfig(
      BuildTargetConfig targetConfig, BuildEnvConfig envConfig);

  /**
   * Build the given buildConfig using the given binary.
   *
   * @return elapsed time of the build
   */
  double buildAndGetElapsedTime(Path buildBinary, ImmutableList<String> args)
      throws CommandException;

  /** Clean the previous build results. */
  void clean() throws CommandException;
}
