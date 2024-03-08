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
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletionException;
import javax.annotation.Nullable;

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

  /** Key for {@link PlatformMappingValue} based on the location of the mapping file. */
  @ThreadSafety.Immutable
  @AutoCodec
  public static final class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

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

    private static Key create(
        PathFragment workspaceRelativeMappingPath, boolean wasExplicitlySetByUser) {
      return interner.intern(new Key(workspaceRelativeMappingPath, wasExplicitlySetByUser));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
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

    boolean wasExplicitlySetByUser() {
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

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  private final ImmutableMap<Label, NativeAndStarlarkFlags> platformsToFlags;
  private final ImmutableMap<ImmutableList<String>, Label> flagsToPlatforms;
  private final ImmutableSet<Class<? extends FragmentOptions>> optionsClasses;
  private final LoadingCache<NativeAndStarlarkFlags, OptionsParsingResult> parserCache;
  private final LoadingCache<BuildOptions, BuildOptions> mappingCache;
  private final RepositoryMapping mainRepositoryMapping;

  /**
   * Creates a new mapping value which will match on the given platforms (if a target platform is
   * set on the key to be mapped), otherwise on the set of flags.
   *
   * @param platformsToFlags mapping from target platform label to the command line style flags that
   *     should be parsed & modified if that platform is set
   * @param flagsToPlatforms mapping from a set of command line style flags to a target platform
   *     that should be set if the flags match the mapped options
   * @param optionsClasses default options classes that should be used for options parsing
   * @param mainRepositoryMapping the main repo mapping used to parse label-valued options
   */
  PlatformMappingValue(
      ImmutableMap<Label, NativeAndStarlarkFlags> platformsToFlags,
      ImmutableMap<ImmutableList<String>, Label> flagsToPlatforms,
      ImmutableSet<Class<? extends FragmentOptions>> optionsClasses,
      RepositoryMapping mainRepositoryMapping) {
    this.platformsToFlags = checkNotNull(platformsToFlags);
    this.flagsToPlatforms = checkNotNull(flagsToPlatforms);
    this.optionsClasses = checkNotNull(optionsClasses);
    this.mainRepositoryMapping = checkNotNull(mainRepositoryMapping);
    this.parserCache =
        Caffeine.newBuilder()
            .initialCapacity(platformsToFlags.size() + flagsToPlatforms.size())
            .build(NativeAndStarlarkFlags::parse);
    this.mappingCache = Caffeine.newBuilder().weakKeys().build(this::computeMapping);
  }

  /**
   * Maps one {@link BuildOptions} to another by way of mappings provided in a file.
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
  public BuildOptions map(BuildOptions original) throws OptionsParsingException {
    try {
      return mappingCache.get(original);
    } catch (CompletionException e) {
      throwIfInstanceOf(e.getCause(), OptionsParsingException.class);
      throwIfUnchecked(e.getCause());
      throw e;
    }
  }

  private BuildOptions computeMapping(BuildOptions originalOptions) throws OptionsParsingException {

    if (originalOptions.hasNoConfig()) {
      // The empty configuration (produced by NoConfigTransition) is terminal: it'll never change.
      return originalOptions;
    }

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
        return originalOptions;
      }

      modifiedOptions =
          originalOptions.applyParsingResult(parseWithCache(platformsToFlags.get(targetPlatform)));
    } else {
      boolean mappingFound = false;
      for (Map.Entry<ImmutableList<String>, Label> flagsToPlatform : flagsToPlatforms.entrySet()) {
        if (originalOptions.matches(
            parseWithCache(
                NativeAndStarlarkFlags.builder()
                    .nativeFlags(flagsToPlatform.getKey())
                    .optionsClasses(this.optionsClasses)
                    .repoMapping(this.mainRepositoryMapping)
                    .build()))) {
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

    return modifiedOptions;
  }

  private OptionsParsingResult parseWithCache(NativeAndStarlarkFlags args)
      throws OptionsParsingException {
    try {
      return parserCache.get(args);
    } catch (CompletionException e) {
      throwIfInstanceOf(e.getCause(), OptionsParsingException.class);
      throwIfUnchecked(e.getCause());
      throw e;
    }
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
