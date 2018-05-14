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

package com.google.devtools.build.lib.bazel.rules;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.List;

/**
 * Provides default {@link BuildOptions} based on the Fragment Classes provided. These default
 * BuildOptions are used to create {@link BuildOptions.OptionsDiffForReconstruction} to store in
 * {@link BuildConfigurationValue.Key}.
 */
public class DefaultBuildOptionsForDiffing {

  private static final ImmutableMap<Class<? extends FragmentOptions>, List<String>>
      DEFAULT_OPTIONS_MAP = ImmutableMap.of();

  public static BuildOptions getDefaultBuildOptionsForFragments(
      List<Class<? extends FragmentOptions>> fragmentClasses) {
    ArrayList<String> collector = new ArrayList<>();
    for (Class<? extends FragmentOptions> clazz : fragmentClasses) {
      List<String> options = DEFAULT_OPTIONS_MAP.get(clazz);
      if (options != null) {
        collector.addAll(options);
      }
    }
    try {
      String[] stringCollector = new String[collector.size()];
      return BuildOptions.of(fragmentClasses, collector.toArray(stringCollector));
    } catch (OptionsParsingException e) {
      throw new IllegalArgumentException("Failed to parse default options", e);
    }
  }
}
