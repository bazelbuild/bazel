// Copyright 2022 The Bazel Authors. All rights reserved.
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

import static java.util.Arrays.stream;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionMetadataTag;

/** Stores information about a build option gathered via reflection. */
public final class OptionInfo {
  private static final LoadingCache<
          ImmutableSet<Class<? extends FragmentOptions>>, ImmutableMap<String, OptionInfo>>
      optionInfoCache =
          CacheBuilder.newBuilder().weakValues().build(CacheLoader.from(OptionInfo::buildMap));

  private final Class<? extends FragmentOptions> optionClass;
  private final OptionDefinition definition;

  private OptionInfo(Class<? extends FragmentOptions> optionClass, OptionDefinition definition) {
    this.optionClass = optionClass;
    this.definition = definition;
  }

  public Class<? extends FragmentOptions> getOptionClass() {
    return optionClass;
  }

  public OptionDefinition getDefinition() {
    return definition;
  }

  public boolean hasOptionMetadataTag(OptionMetadataTag tag) {
    return stream(getDefinition().getOptionMetadataTags()).anyMatch(tag::equals);
  }

  /** For all the options in the BuildOptions, build a map from option name to its information. */
  public static ImmutableMap<String, OptionInfo> buildMapFrom(BuildOptions buildOptions) {
    // Copy the key set so the cache does not retain the option values through a map view.
    return optionInfoCache.getUnchecked(ImmutableSet.copyOf(buildOptions.getFragmentClasses()));
  }

  private static ImmutableMap<String, OptionInfo> buildMap(
      ImmutableSet<Class<? extends FragmentOptions>> optionClasses) {
    ImmutableMap.Builder<String, OptionInfo> builder = new ImmutableMap.Builder<>();

    for (Class<? extends FragmentOptions> optionClass : optionClasses) {
      ImmutableList<? extends OptionDefinition> optionDefinitions =
          OptionDefinition.getOptionDefinitions(optionClass);
      for (OptionDefinition def : optionDefinitions) {
        String optionName = def.getOptionName();
        builder.put(optionName, new OptionInfo(optionClass, def));
      }
    }

    return builder.build();
  }
}
