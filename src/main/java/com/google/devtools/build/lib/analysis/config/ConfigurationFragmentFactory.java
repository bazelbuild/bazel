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
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import javax.annotation.Nullable;

/**
 * A factory that creates configuration fragments.
 */
public interface ConfigurationFragmentFactory {
  /**
   * Creates a configuration fragment.
   *
   * <p>All implementations should override this method unless they have a really good reason
   * to override {@link #create(ConfigurationEnvironment, BuildOptions)} instead. The latter
   * interface is slated for removal once we detach legacy callers.
   *
   * @param buildOptions command-line options (see {@link FragmentOptions})
   * @return the configuration fragment or null if some required dependencies are missing.
   */
  @Nullable
  default BuildConfiguration.Fragment create(BuildOptions buildOptions)
      throws InvalidConfigurationException, InterruptedException {
    throw new IllegalStateException(
        "One of this method's signatures must be overridden to have a valid fragment creator");
  }

  /**
   * Creates a configuration fragment: <b>LEGACY VERSION</b>.
   *
   * <p>For implementations that cannot override {@link #create(BuildOptions)} because they really
   * need access to {@link ConfigurationEnvironment}. {@link ConfigurationEnvironment} adds extra
   * dependencies to fragment creation that makes the whole process more complicated and delicate.
   * We're also working on Bazel enhancements that will make current calls unnecessary. So this
   * version really only exists as a stopgap before we can migrate away the legacy calls.
   *
   * @param env the ConfigurationEnvironment for querying targets and paths
   * @param buildOptions command-line options (see {@link FragmentOptions})
   * @return the configuration fragment or null if some required dependencies are missing.
   */
  @Deprecated
  @Nullable
  default BuildConfiguration.Fragment create(ConfigurationEnvironment env,
      BuildOptions buildOptions) throws InvalidConfigurationException, InterruptedException {
    return create(buildOptions);
  }


  /**
   * @return the exact type of the fragment this factory creates.
   */
  Class<? extends Fragment> creates();

  /**
   * Returns the option classes needed to load this fragment.
   */
  ImmutableSet<Class<? extends FragmentOptions>> requiredOptions();
}
