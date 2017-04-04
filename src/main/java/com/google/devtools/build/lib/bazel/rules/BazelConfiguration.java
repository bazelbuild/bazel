// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Bazel-specific configuration fragment.
 */
@Immutable
public class BazelConfiguration extends Fragment {
  /**
   * Loader for Bazel-specific settings.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException {
      return new BazelConfiguration();
    }

    @Override
    public Class<? extends Fragment> creates() {
      return BazelConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.of();
    }
  }

  public BazelConfiguration() {
  }

  @Override
  public PathFragment getShellExecutable() {
    if (OS.getCurrent() == OS.WINDOWS) {
      String path = System.getenv("BAZEL_SH");
      if (path != null) {
        return PathFragment.create(path);
      } else {
        return PathFragment.create("c:/tools/msys64/usr/bin/bash.exe");
      }
    }
    if (OS.getCurrent() == OS.FREEBSD) {
      String path = System.getenv("BAZEL_SH");
      if (path != null) {
        return  PathFragment.create(path);
      } else {
        return PathFragment.create("/usr/local/bin/bash");
      }
    }
    return PathFragment.create("/bin/bash");
  }

  @Override
  public void setupShellEnvironment(ImmutableMap.Builder<String, String> builder) {
    builder.put("PATH", pathOrDefault(System.getenv("PATH"), getShellExecutable()));

    String ldLibraryPath = System.getenv("LD_LIBRARY_PATH");
    if (ldLibraryPath != null) {
      builder.put("LD_LIBRARY_PATH", ldLibraryPath);
    }

    String tmpdir = System.getenv("TMPDIR");
    if (tmpdir != null) {
      builder.put("TMPDIR", tmpdir);
    }
  }

  private static String pathOrDefault(@Nullable String path, @Nullable PathFragment sh) {
    if (OS.getCurrent() != OS.WINDOWS) {
      return path == null ? "/bin:/usr/bin" : path;
    }

    // Attempt to compute the MSYS root (the real Windows path of "/") from `sh`.
    String newPath = "";
    if (sh != null && sh.getParentDirectory() != null) {
      newPath = sh.getParentDirectory().getPathString();
      if (sh.getParentDirectory().endsWith(PathFragment.create("usr/bin"))) {
        newPath +=
            ";" + sh.getParentDirectory().getParentDirectory().replaceName("bin").getPathString();
      } else if (sh.getParentDirectory().endsWith(PathFragment.create("bin"))) {
        newPath +=
            ";" + sh.getParentDirectory().replaceName("usr").getRelative("bin").getPathString();
      }
      newPath = newPath.replace('/', '\\');

      if (path != null) {
        newPath += ";" + path;
      }
    }
    return newPath;
  }
}
