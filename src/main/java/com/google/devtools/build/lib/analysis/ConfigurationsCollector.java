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
package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;

/** Collects known configurations based on a set of dependencies. */
public interface ConfigurationsCollector {
  /*
   * Retrieves the configurations needed for the given deps. If {@link
   * com.google.devtools.build.lib.analysis.config.CoreOptions#trimConfigurations()} is true, trims
   * their fragments to only those needed by their transitive closures. Else unconditionally
   * includes all fragments.
   *
   * <p>Skips targets with loading phase errors.
   */
  ConfigurationsResult getConfigurations(
      ExtendedEventHandler eventHandler, BuildOptions fromOptions, Iterable<Dependency> keys)
      throws InvalidConfigurationException;
}
