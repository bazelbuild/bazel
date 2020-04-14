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

package com.google.devtools.build.lib.rules.apple.swift;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkbuildapi.apple.SwiftConfigurationApi;

/**
 * A configuration containing flags required for Swift tools. This is used primarily by swift_*
 * family of rules written in Starlark.
 */
@Immutable
public class SwiftConfiguration extends Fragment implements SwiftConfigurationApi {
  private final ImmutableList<String> copts;

  private SwiftConfiguration(SwiftCommandLineOptions options) {
    this.copts = ImmutableList.copyOf(options.copts);
  }

  /** Returns a list of options to use for compiling Swift. */
  @Override
  public ImmutableList<String> getCopts() {
    return copts;
  }

  /** Loads {@link SwiftConfiguration} from build options. */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public SwiftConfiguration create(BuildOptions buildOptions)
        throws InvalidConfigurationException {
      SwiftCommandLineOptions options = buildOptions.get(SwiftCommandLineOptions.class);

      return new SwiftConfiguration(options);
    }

    @Override
    public Class<? extends Fragment> creates() {
      return SwiftConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(SwiftCommandLineOptions.class);
    }
  }
}
