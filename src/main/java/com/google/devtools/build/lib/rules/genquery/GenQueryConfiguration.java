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

package com.google.devtools.build.lib.rules.genquery;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;

/** {@link BuildConfiguration.Fragment} for {@link GenQuery}. */
public class GenQueryConfiguration extends BuildConfiguration.Fragment {

  /** GenQuery-specific options. */
  public static class GenQueryOptions extends FragmentOptions {
    @Option(
        name = "compress_in_memory_genquery_results",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
        help =
            "If true, the in-memory representation of genquery results may be compressed as "
                + "is necessary. Can save sufficient memory at the expense of more CPU usage.")
    public boolean compressInMemoryResults;
  }

  static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(BuildOptions buildOptions) throws InvalidConfigurationException {
      return new GenQueryConfiguration(
          buildOptions.get(GenQueryOptions.class).compressInMemoryResults);
    }

    @Override
    public Class<? extends Fragment> creates() {
      return GenQueryConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.of(GenQueryOptions.class);
    }
  }

  private final boolean inMemoryCompressionEnabled;

  GenQueryConfiguration(boolean inMemoryCompressionEnabled) {
    this.inMemoryCompressionEnabled = inMemoryCompressionEnabled;
  }

  /** Returns whether or not genquery stored in memory can be stored in compressed form. */
  boolean inMemoryCompressionEnabled() {
    return inMemoryCompressionEnabled;
  }
}
