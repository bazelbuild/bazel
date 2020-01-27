// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Class to work with the shell toolchain, e.g. get the shell interpreter's path. */
public final class ShToolchain {

  /**
   * Returns the shell executable's path, or an empty path if not set.
   *
   * <p>This method checks the configuration's {@link ShellConfiguration} fragment.
   */
  public static PathFragment getPath(BuildConfiguration config) {
    PathFragment result = PathFragment.EMPTY_FRAGMENT;

    ShellConfiguration configFragment =
        (ShellConfiguration) config.getFragment(ShellConfiguration.class);
    if (configFragment != null) {
      PathFragment path = configFragment.getShellExecutable();
      if (path != null) {
        result = path;
      }
    }

    return result;
  }

  /**
   * Returns the shell executable's path, or reports a rule error if the path is empty.
   *
   * <p>This method checks the rule's configuration's {@link ShellConfiguration} fragment for the
   * shell executable's path. If null or empty, the method reports an error against the rule.
   */
  public static PathFragment getPathOrError(RuleContext ctx) {
    PathFragment result = getPath(ctx.getConfiguration());

    if (result.isEmpty()) {
      ctx.ruleError(
          "This rule needs a shell interpreter. Use the --shell_executable=<path> flag to specify"
              + " the interpreter's path, e.g. --shell_executable=/usr/local/bin/bash");
    }

    return result;
  }

  private ShToolchain() {}
}
