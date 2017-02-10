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

/** Interface that includes methods for a build case including all build target information. */
interface BuildCase {

  /** Returns a list of build environment configs. */
  ImmutableList<BuildEnvConfig> getBuildEnvConfigs();

  /** Returns a list of build target configs. */
  ImmutableList<BuildTargetConfig> getBuildTargetConfigs();

  /**
   * Returns a list of code versions (can be anything you specified) of {@code builder}.
   */
  ImmutableList<String> getCodeVersions(Builder builder, BenchmarkOptions options)
      throws IOException, CommandException;

  /**
   * Prepares generated code for build.
   *
   * @param copyDir the source code path for copy
   * @param generatedCodePath the path to put generated code
   */
  void prepareGeneratedCode(Path copyDir, Path generatedCodePath) throws IOException;
}
