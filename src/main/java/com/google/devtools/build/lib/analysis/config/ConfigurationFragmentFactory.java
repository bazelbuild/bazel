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
package com.google.devtools.build.lib.analysis.config;

import com.google.common.collect.ImmutableSet;
import javax.annotation.Nullable;

/**
 * A factory that instantiates configuration fragments, and which knows some "static" information
 * about these fragments.
 */
public interface ConfigurationFragmentFactory {
  /**
   * Creates a configuration fragment from the given command-line options.
   *
   * <p>{@code buildOptions} is only guaranteed to hold those {@link FragmentOptions} that are
   * listed by {@link #requiredOptions}.
   *
   * @return the configuration fragment, or null if some required dependencies are missing.
   */
  @Nullable
  Fragment create(BuildOptions buildOptions) throws InvalidConfigurationException;

  /** Returns the exact type of the fragment this factory creates. */
  Class<? extends Fragment> creates();

  /** Returns the option classes needed to create this fragment. */
  ImmutableSet<Class<? extends FragmentOptions>> requiredOptions();
}
