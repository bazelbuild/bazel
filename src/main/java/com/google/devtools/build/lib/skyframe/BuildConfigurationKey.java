// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentClassSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Objects;

/**
 * {@link SkyKey} for {@link com.google.devtools.build.lib.analysis.config.BuildConfigurationValue}.
 */
@AutoCodec
public final class BuildConfigurationKey implements SkyKey {

  /**
   * Creates a new configuration key based on the given options, after applying a platform mapping
   * transformation.
   *
   * @param platformMappingValue sky value that can transform a configuration key based on a
   *     platform mapping
   * @param fragments set of options fragments this configuration should cover
   * @param options the desired configuration
   * @throws OptionsParsingException if the platform mapping cannot be parsed
   */
  public static BuildConfigurationKey withPlatformMapping(
      PlatformMappingValue platformMappingValue, FragmentClassSet fragments, BuildOptions options)
      throws OptionsParsingException {
    return platformMappingValue.map(withoutPlatformMapping(fragments, options));
  }

  /**
   * Returns the key for a requested configuration.
   *
   * <p>Callers are responsible for applying the platform mapping or ascertaining that a platform
   * mapping is not required.
   *
   * @param fragments the fragments the configuration should contain
   * @param options the {@link BuildOptions} object the {@link BuildOptions} should be rebuilt from
   */
  @AutoCodec.Instantiator
  public static BuildConfigurationKey withoutPlatformMapping(
      FragmentClassSet fragments, BuildOptions options) {
    return interner.intern(new BuildConfigurationKey(fragments, options));
  }

  private static final Interner<BuildConfigurationKey> interner = BlazeInterners.newWeakInterner();

  private final FragmentClassSet fragments;
  private final BuildOptions options;
  private final int hashCode;

  private BuildConfigurationKey(FragmentClassSet fragments, BuildOptions options) {
    this.fragments = Preconditions.checkNotNull(fragments);
    this.options = Preconditions.checkNotNull(options);
    this.hashCode = Objects.hash(fragments, options);
  }

  public FragmentClassSet getFragments() {
    return fragments;
  }

  public BuildOptions getOptions() {
    return options;
  }

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.BUILD_CONFIGURATION;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof BuildConfigurationKey)) {
      return false;
    }
    BuildConfigurationKey otherConfig = (BuildConfigurationKey) o;
    return options.equals(otherConfig.options) && fragments.equals(otherConfig.fragments);
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  @Override
  public String toString() {
    // This format is depended on by integration tests.
    // TODO(blaze-configurability-team): This should at least include the length of fragments.
    // to at least remind devs that this Key has TWO key parts.
    return "BuildConfigurationKey[" + options.checksum() + "]";
  }

  /**
   * Returns a string representation that can be safely used for comparison purposes.
   *
   * <p>Unlike toString, which is short and good for printing in debug contexts, this is long
   * because it includes sufficient information in options and fragments. toString alone is
   * insufficient because multiple Keys can have the same options checksum (and thus same toString)
   * but different fragments.
   *
   * <p>This function is meant to address two potential, trimming-related scenarios: 1. If trimming
   * by only trimming BuildOptions (e.g. --trim_test_configuration), then after the initial
   * trimming, fragments has extra classes (corresponding to those trimmed). Notably, dependencies
   * of trimmed targets will create Keys with a properly trimmed set of fragments. Thus, will easily
   * have two Keys with the same (trimmed) BuildOptions but different fragments yet corresponding to
   * the same (trimmed) BuildConfigurationValue.
   *
   * <p>2. If trimming by only trimming fragments (at time of this comment, unsure whether this is
   * ever done in active code), then BuildOptions has extra classes. The returned
   * BuildConfigurationValue is properly trimmed (with the extra classes BuildOptions removed)
   * although notably with a different checksum compared to the Key checksum. Note that given a
   * target that is doing trimming like this, the reverse dependency of the target (i.e. without
   * trimming) could easily involve a Key with the same (untrimmed!) BuildOptions but different
   * fragments. However, unlike in case 1, they will correspond to different
   * BuildConfigurationValue.
   */
  public String toComparableString() {
    return "BuildConfigurationKey[" + options.checksum() + ", " + fragments + "]";
  }
}
