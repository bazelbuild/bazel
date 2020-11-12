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
package com.google.devtools.build.lib.rules.python;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;

/**
 * A factory implementation for {@link PythonConfiguration} objects.
 */
public class PythonConfigurationLoader implements ConfigurationFragmentFactory {
  @Override
  public PythonConfiguration create(BuildOptions buildOptions)
      throws InvalidConfigurationException {
    return new PythonConfiguration(buildOptions);
  }

  @Override
  public Class<? extends Fragment> creates() {
    return PythonConfiguration.class;
  }
}

