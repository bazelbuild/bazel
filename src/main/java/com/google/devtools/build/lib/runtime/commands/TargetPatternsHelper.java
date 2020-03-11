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

package com.google.devtools.build.lib.runtime.commands;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

/** Provides support for reading target patterns from a file or the command-line. */
final class TargetPatternsHelper {

  private TargetPatternsHelper() {}

  /**
   * Reads a list of target patterns, either from the command-line residue or by reading newline
   * delimited target patterns from the --target_pattern_file flag. If --target_pattern_file is
   * specified and options contain a residue, or file cannot be read it throws an exception instead.
   */
  public static List<String> readFrom(CommandEnvironment env, OptionsParsingResult options)
      throws TargetPatternsHelperException {
    List<String> targets = options.getResidue();
    BuildRequestOptions buildRequestOptions = options.getOptions(BuildRequestOptions.class);
    if (!targets.isEmpty() && !buildRequestOptions.targetPatternFile.isEmpty()) {
      throw new TargetPatternsHelperException(
          "Command-line target pattern and --target_pattern_file cannot both be specified");
    } else if (!buildRequestOptions.targetPatternFile.isEmpty()) {
      // Works for absolute or relative file.
      Path residuePath =
          env.getWorkingDirectory().getRelative(buildRequestOptions.targetPatternFile);
      try {
        targets =
            Lists.newArrayList(FileSystemUtils.readLines(residuePath, StandardCharsets.UTF_8));
      } catch (IOException e) {
        throw new TargetPatternsHelperException(
            "I/O error reading from " + residuePath.getPathString() + ": " + e.getMessage());
      }
    } else {
      try (SilentCloseable closeable =
          Profiler.instance().profile("ProjectFileSupport.getTargets")) {
        targets = ProjectFileSupport.getTargets(env.getRuntime().getProjectFileProvider(), options);
      }
    }
    return targets;
  }

  /** Thrown when target patterns were incorrectly specified. */
  public static class TargetPatternsHelperException extends Exception {
    public TargetPatternsHelperException(String message) {
      super(message);
    }
  }
}
