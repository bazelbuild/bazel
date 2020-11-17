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
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.starlarkbuildapi.apple.SwiftConfigurationApi;

/**
 * A configuration containing flags required for Swift tools. This is used primarily by swift_*
 * family of rules written in Starlark.
 */
@Immutable
@RequiresOptions(options = {SwiftCommandLineOptions.class})
public class SwiftConfiguration extends Fragment implements SwiftConfigurationApi {
  private final ImmutableList<String> copts;

  public SwiftConfiguration(BuildOptions buildOptions) {
    this.copts = ImmutableList.copyOf(buildOptions.get(SwiftCommandLineOptions.class).copts);
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  /** Returns a list of options to use for compiling Swift. */
  @Override
  public ImmutableList<String> getCopts() {
    return copts;
  }

  /** Loads {@link SwiftConfiguration} from build options. */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Class<? extends Fragment> creates() {
      return SwiftConfiguration.class;
    }
  }
}
