// Copyright 2017 The Bazel Authors. All rights reserved.
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
// limitations under the License

package com.google.devtools.build.lib.rules.config;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;
import javax.annotation.Nullable;

/**
 * Configuration fragment for Android's config_feature_flag, flags which can be defined in BUILD
 * files.
 */
public final class ConfigFeatureFlagConfiguration extends BuildConfiguration.Fragment {
  /**
   * A configuration fragment loader able to create instances of {@link
   * ConfigFeatureFlagConfiguration} from {@link ConfigFeatureFlagOptions}.
   */
  public static final class Loader implements ConfigurationFragmentFactory {
    @Override
    public BuildConfiguration.Fragment create(BuildOptions buildOptions)
        throws InvalidConfigurationException {
      ConfigFeatureFlagOptions options = buildOptions.get(ConfigFeatureFlagOptions.class);
      if (!options.unknownFlags.isEmpty()) {
        ImmutableList.Builder<String> errorMessage = new ImmutableList.Builder<>();
        for (Label missingLabel : options.unknownFlags) {
          errorMessage.add(
              String.format(
                  "Feature flag %1$s was accessed in a configuration it is not present in. All "
                      + "targets which depend on %1$s directly or indirectly must name it in their "
                      + "transitive_configs attribute.",
                  missingLabel));
        }
        throw new InvalidConfigurationException(
            "Some feature flags were incorrectly specified:\n"
                + Joiner.on("\n").join(errorMessage.build()));
      }

      return new ConfigFeatureFlagConfiguration(options);
    }

    @Override
    public Class<? extends BuildConfiguration.Fragment> creates() {
      return ConfigFeatureFlagConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(ConfigFeatureFlagOptions.class);
    }
  }

  /** Exception thrown when a flag is accessed in a configuration it is not present in. */
  public static final class MissingFlagException extends RuntimeException {
    public MissingFlagException(Label label) {
      super(
          String.format(
              "Feature flag %1$s was accessed in a configuration it was trimmed from.", label));
    }
  }

  private final ImmutableSortedMap<Label, String> flagValues;
  private final ImmutableSortedSet<Label> knownDefaultFlags;
  private final boolean isTrimmed;
  @Nullable private final String flagHash;

  /** Creates a new configuration fragment from the given {@link ConfigFeatureFlagOptions}. */
  @VisibleForTesting
  ConfigFeatureFlagConfiguration(ConfigFeatureFlagOptions options) {
    // TODO(mstaib): we'd love to only construct these when we're trimmed, but this is still what
    // the top level looks like - make constructing an untrimmed configuration an error when no
    // configurations are constructed untrimmed.
    this.flagValues = options.getFlagValues();
    this.knownDefaultFlags = options.getKnownDefaultFlags().orElse(ImmutableSortedSet.of());
    this.isTrimmed = options.enforceTransitiveConfigsForConfigFeatureFlag && options.isTrimmed();
    // We don't hash flags set to their default values; all valid configurations of a target have
    // the same set of known flags, so the set of flags set to something other than their default
    // values is enough to disambiguate configurations. Similarly, isTrimmed need not be hashed;
    // enforceTransitiveConfigsForConfigFeatureFlag should not change within a build, and when it's
    // enabled, the only configuration which is untrimmed (the top-level configuration) shouldn't
    // be used for any actual targets.
    this.flagHash = flagValues.isEmpty() ? null : hashFlags(flagValues);
  }

  /** Converts the given flag values into a string hash for use as an output directory fragment. */
  private static String hashFlags(SortedMap<Label, String> flagValues) {
    // This hash function is relatively fast and stable between JVM invocations.
    Hasher hasher = Hashing.murmur3_128().newHasher();

    for (Map.Entry<Label, String> flag : flagValues.entrySet()) {
      hasher.putUnencodedChars(flag.getKey().toString());
      hasher.putByte((byte) 0);
      hasher.putUnencodedChars(flag.getValue());
      hasher.putByte((byte) 0);
    }
    return hasher.hash().toString();
  }

  /**
   * Retrieves the value of a configuration flag.
   *
   * <p>If the flag is not set in the current configuration, then the returned value will be absent.
   *
   * <p>If the flag has been trimmed from the current configuration, a RuntimeException
   * (MissingFlagException) will be thrown. Because the configuration should fail to construct if a
   * required flag is missing, and because config_feature_flag (the only intended user of this
   * method) automatically requires itself, this should not come to pass.
   *
   * <p>This method should only be used by the rule whose label is passed here. Other rules should
   * depend on that rule and read a provider exported by it. To encourage callers of this method to
   * do the right thing, this class takes {@link ArtifactOwner} instead of {@link Label}; to get the
   * ArtifactOwner for a rule, call {@code ruleContext.getOwner()}.
   */
  public Optional<String> getFeatureFlagValue(ArtifactOwner owner) {
    if (flagValues.containsKey(owner.getLabel())) {
      return Optional.of(flagValues.get(owner.getLabel()));
    } else if (!isTrimmed || knownDefaultFlags.contains(owner.getLabel())) {
      return Optional.empty();
    } else {
      throw new MissingFlagException(owner.getLabel());
    }
  }

  /**
   * Returns a fragment of the output directory name for this configuration, based on the set of
   * flags and their values. It will be {@code null} if no flags are set.
   */
  @Nullable
  @Override
  public String getOutputDirectoryName() {
    return flagHash;
  }
}
