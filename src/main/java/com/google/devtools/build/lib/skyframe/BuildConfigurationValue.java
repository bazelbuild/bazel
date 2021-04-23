// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentClassSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.Serializable;
import java.util.Objects;
import java.util.Set;

/** A Skyframe value representing a {@link BuildConfiguration}. */
// TODO(bazel-team): mark this immutable when BuildConfiguration is immutable.
// @Immutable
@AutoCodec
@ThreadSafe
public class BuildConfigurationValue implements SkyValue {
  private final BuildConfiguration configuration;

  BuildConfigurationValue(BuildConfiguration configuration) {
    this.configuration = configuration;
  }

  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  /**
   * Creates a new configuration key based on the given options, after applying a platform mapping
   * transformation.
   *
   * @param platformMappingValue sky value that can transform a configuration key based on a
   *     platform mapping
   * @param defaultBuildOptions set of native build options without modifications based on parsing
   *     flags
   * @param fragments set of options fragments this configuration should cover
   * @param options the desired configuration
   * @throws OptionsParsingException if the platform mapping cannot be parsed
   */
  public static Key keyWithPlatformMapping(
      PlatformMappingValue platformMappingValue,
      BuildOptions defaultBuildOptions,
      FragmentClassSet fragments,
      BuildOptions options)
      throws OptionsParsingException {
    return platformMappingValue.map(
        keyWithoutPlatformMapping(fragments, options), defaultBuildOptions);
  }

  /**
   * Creates a new configuration key based on the given options, after applying a platform mapping
   * transformation.
   *
   * @param platformMappingValue sky value that can transform a configuration key based on a
   *     platform mapping
   * @param defaultBuildOptions set of native build options without modifications based on parsing
   *     flags
   * @param fragments set of options fragments this configuration should cover
   * @param options the desired configuration
   * @throws OptionsParsingException if the platform mapping cannot be parsed
   */
  public static Key keyWithPlatformMapping(
      PlatformMappingValue platformMappingValue,
      BuildOptions defaultBuildOptions,
      Set<Class<? extends Fragment>> fragments,
      BuildOptions options)
      throws OptionsParsingException {
    return platformMappingValue.map(
        keyWithoutPlatformMapping(fragments, options), defaultBuildOptions);
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
  @ThreadSafe
  static Key keyWithoutPlatformMapping(
      Set<Class<? extends Fragment>> fragments, BuildOptions options) {
    return Key.create(
        FragmentClassSet.of(
            ImmutableSortedSet.copyOf(BuildConfiguration.lexicalFragmentSorter, fragments)),
        options);
  }

  private static Key keyWithoutPlatformMapping(
      FragmentClassSet fragmentClassSet, BuildOptions options) {
    return Key.create(fragmentClassSet, options);
  }

  /**
   * Returns a configuration key for the given configuration.
   *
   * <p>Note that this key creation method does not apply a platform mapping, it is assumed that the
   * passed configuration was created with one such and thus its key does not need to be mapped
   * again.
   *
   * @param buildConfiguration configuration whose key is requested
   */
  public static Key key(BuildConfiguration buildConfiguration) {
    return keyWithoutPlatformMapping(
        buildConfiguration.fragmentClasses(), buildConfiguration.getOptions());
  }

  /** {@link SkyKey} for {@link BuildConfigurationValue}. */
  @AutoCodec
  public static final class Key implements SkyKey, Serializable {
    private static final Interner<Key> keyInterner = BlazeInterners.newWeakInterner();

    private final FragmentClassSet fragments;
    private final BuildOptions options;
    private final int hashCode;

    @AutoCodec.Instantiator
    @VisibleForSerialization
    static Key create(FragmentClassSet fragments, BuildOptions options) {
      return keyInterner.intern(new Key(fragments, options));
    }

    private Key(FragmentClassSet fragments, BuildOptions options) {
      this.fragments = Preconditions.checkNotNull(fragments);
      this.options = Preconditions.checkNotNull(options);
      this.hashCode = Objects.hash(fragments, options);
    }

    @VisibleForTesting
    public ImmutableSortedSet<Class<? extends Fragment>> getFragments() {
      return fragments.fragmentClasses();
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
      if (!(o instanceof Key)) {
        return false;
      }
      Key otherConfig = (Key) o;
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
      return "BuildConfigurationValue.Key[" + options.checksum() + "]";
    }

    /**
     * Return a string representation that can be safely used for comparison purposes.
     *
     * <p>Unlike toString, which is short and good for printing in debug contexts, this is long
     * because it includes sufficient information in options and fragments. toString alone is
     * insufficient because multiple Keys can have the same options checksum (and thus same
     * toString) but different fragments.
     *
     * <p>This function is meant to address two potential, trimming-related scenarios: 1. If
     * trimming by only trimming BuildOptions (e.g. --trim_test_configuration), then after the
     * initial trimming, fragments has extra classes (corresponding to those trimmed). Notably,
     * dependencies of trimmed targets will create Keys with a properly trimmed set of fragments.
     * Thus, will easily have two Keys with the same (trimmed) BuildOptions but different fragments
     * yet corresponding to the same (trimmed) BuildConfigurationValue.
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
      return "BuildConfigurationValue.Key[" + options.checksum() + ", " + fragments + "]";
    }
  }
}
