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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.MoreObjects;
import com.google.common.base.Throwables;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.UncheckedExecutionException;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/**
 * Stores contents of a platforms/flags mapping file for transforming one {@link
 * BuildConfigurationValue.Key} into another.
 *
 * <p>See <a href=https://docs.google.com/document/d/1Vg_tPgiZbSrvXcJ403vZVAGlsWhH9BUDrAxMOYnO0Ls>
 * the design</a> for more details on how the mapping can be defined and the desired logic on how it
 * is applied to configuration keys.
 */
@AutoCodec
public final class PlatformMappingValue implements SkyValue {

  /** Key for {@link PlatformMappingValue} based on the location of the mapping file. */
  @ThreadSafety.Immutable
  @AutoCodec
  public static final class Key implements SkyKey {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    /**
     * Creates a new platform mappings key with the given, main workspace-relative path to the
     * mappings file, typically derived from the {@code --platform_mappings} flag.
     *
     * <p>If the path is {@code null} the {@link PlatformOptions#DEFAULT_PLATFORM_MAPPINGS default
     * path} will be used and the key marked as not having been set by a user.
     *
     * @param workspaceRelativeMappingPath main workspace relative path to the mappings file or
     *     {@code null} if the default location should be used
     */
    public static Key create(@Nullable PathFragment workspaceRelativeMappingPath) {
      if (workspaceRelativeMappingPath == null) {
        return create(PlatformOptions.DEFAULT_PLATFORM_MAPPINGS, false);
      } else {
        return create(workspaceRelativeMappingPath, true);
      }
    }

    @AutoCodec.Instantiator
    @AutoCodec.VisibleForSerialization
    static Key create(PathFragment workspaceRelativeMappingPath, boolean wasExplicitlySetByUser) {
      return interner.intern(new Key(workspaceRelativeMappingPath, wasExplicitlySetByUser));
    }

    private final PathFragment path;
    private final boolean wasExplicitlySetByUser;

    private Key(PathFragment path, boolean wasExplicitlySetByUser) {
      this.path = path;
      this.wasExplicitlySetByUser = wasExplicitlySetByUser;
    }

    /** Returns the main-workspace relative path this mapping's mapping file can be found at. */
    public PathFragment getWorkspaceRelativeMappingPath() {
      return path;
    }

    public boolean wasExplicitlySetByUser() {
      return wasExplicitlySetByUser;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PLATFORM_MAPPING;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      Key key = (Key) o;
      return Objects.equals(path, key.path) && wasExplicitlySetByUser == key.wasExplicitlySetByUser;
    }

    @Override
    public int hashCode() {
      return Objects.hash(path, wasExplicitlySetByUser);
    }

    @Override
    public String toString() {
      return "PlatformMappingValue.Key{path="
          + path
          + ", wasExplicitlySetByUser="
          + wasExplicitlySetByUser
          + "}";
    }
  }

  private final ImmutableMap<Label, ImmutableSet<String>> platformsToFlags;
  private final ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms;
  private final ImmutableList<Class<? extends FragmentOptions>> optionsClasses;
  private final LoadingCache<ImmutableSet<String>, OptionsParsingResult> parserCache;
  private final LoadingCache<BuildConfigurationValue.Key, BuildConfigurationValue.Key> mappingCache;

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
      ImmutableMap<Label, ImmutableSet<String>> platformsToFlags,
      ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms,
      ImmutableList<Class<? extends FragmentOptions>> optionsClasses) {
    this.platformsToFlags = checkNotNull(platformsToFlags);
    this.flagsToPlatforms = checkNotNull(flagsToPlatforms);
    this.optionsClasses = checkNotNull(optionsClasses);
    this.parserCache =
        CacheBuilder.newBuilder()
            .initialCapacity(platformsToFlags.size() + flagsToPlatforms.size())
            .build(
                new CacheLoader<ImmutableSet<String>, OptionsParsingResult>() {
                  @Override
                  public OptionsParsingResult load(ImmutableSet<String> args)
                      throws OptionsParsingException {
                    return parse(args);
                  }
                });
    this.mappingCache =
        CacheBuilder.newBuilder()
            .weakKeys()
            .build(
                new CacheLoader<BuildConfigurationValue.Key, BuildConfigurationValue.Key>() {
                  @Override
                  public BuildConfigurationValue.Key load(BuildConfigurationValue.Key original)
                      throws OptionsParsingException {
                    return computeMapping(original);
                  }
                });
  }

  /**
   * Maps one {@link BuildConfigurationValue.Key} to another by way of mappings provided in a file.
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
   * @return the mapped key if any mapping matched the original or else the original
   * @throws OptionsParsingException if any of the user configured flags cannot be parsed
   * @throws IllegalArgumentException if the original does not contain a {@link PlatformOptions}
   *     fragment
   */
  public BuildConfigurationValue.Key map(BuildConfigurationValue.Key original)
      throws OptionsParsingException {
    try {
      return mappingCache.get(original);
    } catch (ExecutionException | UncheckedExecutionException e) {
      Throwables.propagateIfPossible(e.getCause(), OptionsParsingException.class);
      throw new IllegalStateException(e);
    }
  }

  private BuildConfigurationValue.Key computeMapping(BuildConfigurationValue.Key original)
      throws OptionsParsingException {
    BuildOptions originalOptions = original.getOptions();

    checkArgument(
        originalOptions.contains(PlatformOptions.class),
        "When using platform mappings, all configurations must contain platform options");

    BuildOptions modifiedOptions = null;

    if (!originalOptions.get(PlatformOptions.class).platforms.isEmpty()) {
      List<Label> platforms = originalOptions.get(PlatformOptions.class).platforms;

      // Platform mapping only supports a single target platform, others are ignored.
      Label targetPlatform = Iterables.getFirst(platforms, null);
      if (!platformsToFlags.containsKey(targetPlatform)) {
        // This can happen if the user has set the platform and any other flags that would normally
        // be mapped from it on the command line instead of relying on the mapping.
        return original;
      }

      modifiedOptions =
          originalOptions.applyParsingResult(parseWithCache(platformsToFlags.get(targetPlatform)));
    } else {
      boolean mappingFound = false;
      for (Map.Entry<ImmutableSet<String>, Label> flagsToPlatform : flagsToPlatforms.entrySet()) {
        if (originalOptions.matches(parseWithCache(flagsToPlatform.getKey()))) {
          modifiedOptions = originalOptions.clone();
          modifiedOptions.get(PlatformOptions.class).platforms =
              ImmutableList.of(flagsToPlatform.getValue());
          mappingFound = true;
          break;
        }
      }

      if (!mappingFound) {
        Label targetPlatform = originalOptions.get(PlatformOptions.class).computeTargetPlatform();
        modifiedOptions = originalOptions.clone();
        modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(targetPlatform);
      }
    }

    return BuildConfigurationValue.keyWithoutPlatformMapping(
        original.getFragments(), modifiedOptions);
  }

  private OptionsParsingResult parseWithCache(ImmutableSet<String> args)
      throws OptionsParsingException {
    try {
      return parserCache.get(args);
    } catch (ExecutionException e) {
      Throwables.propagateIfPossible(e.getCause(), OptionsParsingException.class);
      throw new IllegalStateException(e);
    }
  }

  private OptionsParsingResult parse(Iterable<String> args) throws OptionsParsingException {
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(optionsClasses)
            // We need the ability to re-map internal options in the mappings file.
            .ignoreInternalOptions(false)
            .build();
    parser.parse(ImmutableList.copyOf(args));
    // TODO(schmitt): Parse starlark options as well.
    return parser;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof PlatformMappingValue)) {
      return false;
    }
    PlatformMappingValue that = (PlatformMappingValue) obj;
    return this.flagsToPlatforms.equals(that.flagsToPlatforms)
        && this.platformsToFlags.equals(that.platformsToFlags)
        && this.optionsClasses.equals(that.optionsClasses);
  }

  @Override
  public int hashCode() {
    return Objects.hash(flagsToPlatforms, platformsToFlags, optionsClasses);
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
