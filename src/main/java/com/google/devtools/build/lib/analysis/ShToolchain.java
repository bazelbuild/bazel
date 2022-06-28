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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Class to work with the shell toolchain, e.g. get the shell interpreter's path. */
public final class ShToolchain {

  private static PathFragment getHostOrDefaultPath() {
    OS current = OS.getCurrent();
    if (!ShellConfiguration.getShellExecutables().containsKey(current)) {
      current = OS.UNKNOWN;
    }
    Preconditions.checkState(
        ShellConfiguration.getShellExecutables().containsKey(current),
        "shellExecutableFinder should set a value with key '%s'",
        current);

    return ShellConfiguration.getShellExecutables().get(current);
  }

  /**
   * Returns the default shell executable's path for the host OS.
   *
   * <p>This method checks the configuration's {@link ShellConfiguration} fragment.
   */
  public static PathFragment getPathForHost(BuildConfigurationValue config) {
    ShellConfiguration configFragment = config.getFragment(ShellConfiguration.class);
    if (configFragment != null) {
      if (configFragment.getOptionsBasedDefault() != null) {
        return configFragment.getOptionsBasedDefault();
      } else {
        return getHostOrDefaultPath();
      }
    }
    return PathFragment.EMPTY_FRAGMENT;
  }

  /**
   * Returns the shell executable's path for the provided platform. If none is present, return the
   * path for the host platform. Otherwise, return the default.
   */
  public static PathFragment getPathOrError(PlatformInfo platformInfo) {
    for (OS os : ShellConfiguration.getShellExecutables().keySet()) {
      if (platformInfo
          .constraints()
          .hasConstraintValue(ShellConfiguration.OS_TO_CONSTRAINTS.get(os))) {
        return ShellConfiguration.getShellExecutables().get(os);
      }
    }
    return getHostOrDefaultPath();
  }

  private ShToolchain() {}
}
