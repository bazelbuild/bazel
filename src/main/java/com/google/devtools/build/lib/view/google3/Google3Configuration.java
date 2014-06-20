// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view.google3;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.blaze.BlazeDirectories;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.view.config.BuildOptions;
import com.google.devtools.build.lib.view.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.view.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.view.config.FragmentOptions;
import com.google.devtools.build.lib.view.config.InvalidConfigurationException;
import com.google.devtools.common.options.Option;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Google-specific settings for Blaze.
 */
public class Google3Configuration extends Fragment {
  /**
   * Loader for Google-specific settings.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(ConfigurationEnvironment env, BlazeDirectories directories,
        BuildOptions buildOptions, Map<Class<? extends Fragment>, Fragment> requiredFragments)
        throws InvalidConfigurationException {
      return new Google3Configuration();
    }

    @Override
    public Class<? extends Fragment> creates() {
      return Google3Configuration.class;
    }

    @Override
    public List<Class<? extends Fragment>> requires() {
      return ImmutableList.<Class<? extends Fragment>>of();
    }
  }

  public Google3Configuration() {
  }

  @Override
  public String getName() {
    return "Google";
  }

  @Override
  public String cacheKey() {
    return "";
  }

  @Override
  public void defineExecutables(ImmutableMap.Builder<String, PathFragment> builder) {
    builder.put("sh", new PathFragment("/bin/bash"));
  }
}
