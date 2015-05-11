// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;

import java.util.Collections;
import java.util.Set;

/**
 * A J2ObjC transpiler configuration containing J2ObjC translation flags.
 */
public class J2ObjcConfiguration extends Fragment {
  /**
   * Always-on flags for J2ObjC translation. These flags will always be used for invoking the J2ObjC
   * transpiler. See https://github.com/google/j2objc/wiki/j2objc for flag documentation.
   */
  private static final Set<String> J2OBJC_ALWAYS_ON_TRANSLATION_FLAGS = ImmutableSet.of(
      "--doc-comments",
      "--extract-unsequenced",
      "--final-methods-as-functions",
      "--hide-private-members",
      "--segmented-headers",
      "-XcombineJars");

  /**
   * Allowed flags for J2ObjC translation. See https://github.com/google/j2objc/wiki/j2objc for flag
   * documentation.
   */
  static final Set<String> J2OBJC_BLACKLISTED_TRANSLATION_FLAGS =
      ImmutableSet.of("--prefixes", "--prefix", "-x");

  static final String INVALID_TRANSLATION_FLAGS_MSG_TEMPLATE =
      "J2Objc translation flags: %s not supported. Unsupported flags are: %s";

  /**
   * Configuration loader for {@link J2ObjcConfiguration}.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException {
      return new J2ObjcConfiguration(buildOptions.get(J2ObjcCommandLineOptions.class));
    }

    @Override
    public Class<? extends Fragment> creates() {
      return J2ObjcConfiguration.class;
    }
  }

  private Iterable<String> translationFlags;
  private String cacheKey;

  J2ObjcConfiguration(J2ObjcCommandLineOptions j2ObjcOptions) throws InvalidConfigurationException {
    Set<String> translationFlags = ImmutableSet.<String>builder()
        .addAll(j2ObjcOptions.translationFlags)
        .addAll(J2OBJC_ALWAYS_ON_TRANSLATION_FLAGS)
        .build();

    if (Collections.disjoint(translationFlags, J2OBJC_BLACKLISTED_TRANSLATION_FLAGS)) {
      this.translationFlags = translationFlags;
      this.cacheKey = Joiner.on(" ").join(this.translationFlags);
    } else {
      throw new InvalidConfigurationException(String.format(INVALID_TRANSLATION_FLAGS_MSG_TEMPLATE,
          Joiner.on(",").join(translationFlags),
          Joiner.on(",").join(J2OBJC_BLACKLISTED_TRANSLATION_FLAGS)));
    }
  }

  /**
   * Returns the translation flags used to invoke the J2ObjC transpiler. The returned flags contain
   * both the always-on flags from {@link #J2OBJC_ALWAYS_ON_TRANSLATION_FLAGS} and user-specified
   * flags from {@link J2ObjcCommandLineOptions}. The set of valid flags can be found at
   * {@link #J2OBJC_BLACKLISTED_TRANSLATION_FLAGS}.
   */
  public Iterable<String> getTranslationFlags() {
    return translationFlags;
  }

  @Override
  public String getName() {
    return "J2ObjC";
  }

  @Override
  public String cacheKey() {
    return cacheKey;
  }
}
