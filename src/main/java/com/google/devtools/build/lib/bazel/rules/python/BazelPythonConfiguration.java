// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.python;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.common.options.Option;

import javax.annotation.Nullable;

/**
 * Bazel-specific Python configuration.
 */
@Immutable
public class BazelPythonConfiguration extends BuildConfiguration.Fragment {

  /**
   * Bazel-specific Python configuration options.
   */
  public static final class Options extends FragmentOptions {
    @Option(name = "python2_path",
      defaultValue = "python",
      category = "version",
      help = "Local path to the Python2 executable.")
    public String python2Path;

    @Option(name = "python3_path",
      defaultValue = "python3",
      category = "version",
      help = "Local path to the Python3 executable.")
    public String python3Path;
  }

  /**
   * Loader for the Bazel-specific Python configuration.
   */
  public static final class Loader implements ConfigurationFragmentFactory {
    @Nullable
    @Override
    public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException {
      return new BazelPythonConfiguration(buildOptions.get(Options.class));
    }

    @Override
    public Class<? extends Fragment> creates() {
      return BazelPythonConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(Options.class);
    }
  }

  private final Options options;

  private BazelPythonConfiguration(Options options) {
    this.options = options;
  }

  public String getPython2Path() {
    return options.python2Path;
  }

  public String getPython3Path() {
    return options.python3Path;
  }
}
