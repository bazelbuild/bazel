// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.config;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.base.Throwables.throwIfUnchecked;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.HashCodes;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletionException;

/**
 * Stores contents of a platforms/flags mapping file for transforming one {@link BuildOptions} into
 * another.
 *
 * <p>See <a href=https://docs.google.com/document/d/1Vg_tPgiZbSrvXcJ403vZVAGlsWhH9BUDrAxMOYnO0Ls>
 * the design</a> for more details on how the mapping can be defined and the desired logic on how it
 * is applied to configuration keys.
 */
@AutoCodec
public final class PlatformMappingValue implements SkyValue {

  private final ImmutableMap<Label, ParsedFlagsValue> platformsToFlags;
  private final ImmutableMap<ParsedFlagsValue, Label> flagsToPlatforms;
  private final ImmutableSet<Class<? extends FragmentOptions>> optionsClasses;
  private final LoadingCache<BuildOptions, BuildConfigurationKey> mappingCache;

  /**
   * Creates a new mapping value which will match on the given platforms (if a target platform is
   * set on the key to be mapped), otherwise on the set of flags.
   *
   * @param platformsToFlags mapping from target platform label to the command line style flags that
   *     should be parsed & modified if that platform is set
   * @param flagsToPlatforms mapping from a set of command line style flags to a target platform
   *     that should be set if the flags match the mapped options
   * @param optionsClasses default options classes that should be used for options parsing
   */
  PlatformMappingValue(
      ImmutableMap<Label, ParsedFlagsValue> platformsToFlags,
      ImmutableMap<ParsedFlagsValue, Label> flagsToPlatforms,
      ImmutableSet<Class<? extends FragmentOptions>> optionsClasses) {
    this.platformsToFlags = checkNotNull(platformsToFlags);
    this.flagsToPlatforms = checkNotNull(flagsToPlatforms);
    this.optionsClasses = checkNotNull(optionsClasses);
    this.mappingCache = Caffeine.newBuilder().weakKeys().build(this::computeMapping);
  }

  /**
   * Maps one {@link BuildOptions} to another's {@link BuildConfigurationKey} by way of mappings
   * provided in a file.
   *
   * <p>Returns a {@link BuildConfigurationKey} instead of just {@link BuildOptions} so that caching
   * of mappings also saves the CPU cost of interning {@link BuildConfigurationKey} (which is what
   * callers typically need).
   *
   * <p>The <a href=https://docs.google.com/document/d/1Vg_tPgiZbSrvXcJ403vZVAGlsWhH9BUDrAxMOYnO0Ls>
   * full design</a> contains the details for the mapping logic but in short:
   *
   * <ol>
   *   <li>If a target platform is set on the original then mappings from platform to flags will be
   *       applied.
   *   <li>If no target platform is set then mappings from flags to platforms will be applied.
   *   <li>If no matching flags to platforms mapping was found, the default target platform will be
   *       used.
   * </ol>
   *
   * @param original the key representing the configuration to be mapped
   * @return a {@link BuildConfigurationKey} to request the mapped configuration
   * @throws OptionsParsingException if any of the user configured flags cannot be parsed
   * @throws IllegalArgumentException if the original does not contain a {@link PlatformOptions}
   *     fragment
   */
  public BuildConfigurationKey map(BuildOptions original) throws OptionsParsingException {
    try {
      return mappingCache.get(original);
    } catch (CompletionException e) {
      throwIfInstanceOf(e.getCause(), OptionsParsingException.class);
      throwIfUnchecked(e.getCause());
      throw e;
    }
  }

  private BuildConfigurationKey computeMapping(BuildOptions originalOptions)
      throws OptionsParsingException {
    if (originalOptions.hasNoConfig()) {
      // The empty configuration (produced by NoConfigTransition) is terminal: it'll never change.
      return BuildConfigurationKey.create(originalOptions);
    }

    var platformOptions = originalOptions.get(PlatformOptions.class);
    checkArgument(
        platformOptions != null,
        "When using platform mappings, all configurations must contain platform options");

    if (!platformOptions.platforms.isEmpty()) {
      List<Label> platforms = platformOptions.platforms;

      // Platform mapping only supports a single target platform, others are ignored.
      Label targetPlatform = Iterables.getFirst(platforms, null);
      if (!platformsToFlags.containsKey(targetPlatform)) {
        // This can happen if the user has set the platform and any other flags that would normally
        // be mapped from it on the command line instead of relying on the mapping.
        return BuildConfigurationKey.create(originalOptions);
      }

      ParsedFlagsValue parsedFlags = platformsToFlags.get(targetPlatform);
      return parsedFlags.mergeWith(originalOptions);
    }

    for (Map.Entry<ParsedFlagsValue, Label> flagsToPlatform : flagsToPlatforms.entrySet()) {
      ParsedFlagsValue parsedFlags = flagsToPlatform.getKey();
      Label platformLabel = flagsToPlatform.getValue();
      if (originalOptions.matches(parsedFlags.parsingResult())) {
        BuildOptions modifiedOptions = originalOptions.clone();
        modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(platformLabel);
        return BuildConfigurationKey.create(modifiedOptions);
      }
    }

    // No mapping found.
    Label targetPlatform = platformOptions.computeTargetPlatform();
    BuildOptions modifiedOptions = originalOptions.clone();
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(targetPlatform);
    return BuildConfigurationKey.create(modifiedOptions);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof PlatformMappingValue that)) {
      return false;
    }
    return this.flagsToPlatforms.equals(that.flagsToPlatforms)
        && this.platformsToFlags.equals(that.platformsToFlags)
        && this.optionsClasses.equals(that.optionsClasses);
  }

  @Override
  public int hashCode() {
    return HashCodes.hashObjects(flagsToPlatforms, platformsToFlags, optionsClasses);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("flagsToPlatforms", flagsToPlatforms)
        .add("platformsToFlags", platformsToFlags)
        .add("optionsClasses", optionsClasses)
        .toString();
  }
}
