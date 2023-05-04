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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.PlatformMappingValue.NativeAndStarlarkFlags;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link PlatformMappingValue}. */
@RunWith(JUnit4.class)
public final class PlatformMappingValueTest {

  private static final ImmutableSet<Class<? extends FragmentOptions>>
      BUILD_CONFIG_PLATFORM_OPTIONS = ImmutableSet.of(CoreOptions.class, PlatformOptions.class);

  private static final Label PLATFORM1 = Label.parseCanonicalUnchecked("//platforms:one");
  private static final Label PLATFORM2 = Label.parseCanonicalUnchecked("//platforms:two");

  private static final BuildOptions DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS =
      BuildOptions.getDefaultBuildOptionsForFragments(BUILD_CONFIG_PLATFORM_OPTIONS);
  private static final Label DEFAULT_TARGET_PLATFORM =
      Label.parseCanonicalUnchecked("@local_config_platform//:host");

  @Test
  public void testMapNoMappings() throws OptionsParsingException {
    PlatformMappingValue mappingValue =
        new PlatformMappingValue(
            ImmutableMap.of(), ImmutableMap.of(), BUILD_CONFIG_PLATFORM_OPTIONS);

    BuildConfigurationKey key =
        BuildConfigurationKey.withoutPlatformMapping(DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS);

    BuildConfigurationKey mapped = mappingValue.map(key);

    assertThat(mapped.getOptions().get(PlatformOptions.class).platforms)
        .containsExactly(DEFAULT_TARGET_PLATFORM);
  }

  @Test
  public void testMapPlatformToFlags() throws Exception {
    ImmutableMap<Label, NativeAndStarlarkFlags> platformsToFlags =
        ImmutableMap.of(
            PLATFORM1,
            NativeAndStarlarkFlags.create(
                ImmutableSet.of("--cpu=one", "--compilation_mode=dbg"), ImmutableMap.of()));

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(
            platformsToFlags, ImmutableMap.of(), BUILD_CONFIG_PLATFORM_OPTIONS);

    BuildOptions modifiedOptions = DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.clone();
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM1);

    BuildConfigurationKey mapped = mappingValue.map(keyForOptions(modifiedOptions));

    assertThat(mapped.getOptions().get(CoreOptions.class).cpu).isEqualTo("one");
  }

  @Test
  public void testMapFlagsToPlatform() throws Exception {
    ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms =
        ImmutableMap.of(ImmutableSet.of("--cpu=one", "--compilation_mode=dbg"), PLATFORM1);

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(
            ImmutableMap.of(), flagsToPlatforms, BUILD_CONFIG_PLATFORM_OPTIONS);

    BuildOptions modifiedOptions = DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.clone();
    modifiedOptions.get(CoreOptions.class).cpu = "one";
    modifiedOptions.get(CoreOptions.class).compilationMode = CompilationMode.DBG;

    BuildConfigurationKey mapped = mappingValue.map(keyForOptions(modifiedOptions));

    assertThat(mapped.getOptions().get(PlatformOptions.class).platforms).containsExactly(PLATFORM1);
  }

  @Test
  public void testMapFlagsToPlatformPriority() throws Exception {
    ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms =
        ImmutableMap.of(
            ImmutableSet.of("--cpu=foo", "--compilation_mode=dbg"), PLATFORM1,
            ImmutableSet.of("--cpu=foo"), PLATFORM2);

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(
            ImmutableMap.of(), flagsToPlatforms, BUILD_CONFIG_PLATFORM_OPTIONS);

    BuildOptions modifiedOptions = DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.clone();
    modifiedOptions.get(CoreOptions.class).cpu = "foo";

    BuildConfigurationKey mapped = mappingValue.map(keyForOptions(modifiedOptions));

    assertThat(mapped.getOptions().get(PlatformOptions.class).platforms).containsExactly(PLATFORM2);
  }

  @Test
  public void testMapFlagsToPlatformNoneMatching() throws Exception {
    ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms =
        ImmutableMap.of(ImmutableSet.of("--cpu=foo", "--compilation_mode=dbg"), PLATFORM1);

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(
            ImmutableMap.of(), flagsToPlatforms, BUILD_CONFIG_PLATFORM_OPTIONS);

    BuildOptions modifiedOptions = DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.clone();
    modifiedOptions.get(CoreOptions.class).cpu = "bar";

    BuildConfigurationKey mapped = mappingValue.map(keyForOptions(modifiedOptions));

    assertThat(mapped.getOptions().get(PlatformOptions.class).platforms)
        .containsExactly(DEFAULT_TARGET_PLATFORM);
  }

  @Test
  public void testMapNoPlatformOptions() throws Exception {
    ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms =
        ImmutableMap.of(ImmutableSet.of("--cpu=one"), PLATFORM1);

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(
            ImmutableMap.of(), flagsToPlatforms, BUILD_CONFIG_PLATFORM_OPTIONS);

    BuildOptions options = BuildOptions.of(ImmutableList.of());

    assertThrows(IllegalArgumentException.class, () -> mappingValue.map(keyForOptions(options)));
  }

  @Test
  public void testMapNoMappingIfPlatformIsSetButNotMatching() throws Exception {
    ImmutableMap<Label, NativeAndStarlarkFlags> platformsToFlags =
        ImmutableMap.of(
            PLATFORM1,
            NativeAndStarlarkFlags.create(
                ImmutableSet.of("--cpu=one", "--compilation_mode=dbg"), ImmutableMap.of()));
    ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms =
        ImmutableMap.of(ImmutableSet.of("--cpu=one"), PLATFORM1);

    BuildOptions modifiedOptions = DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.clone();
    modifiedOptions.get(CoreOptions.class).cpu = "one";
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM2);

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(platformsToFlags, flagsToPlatforms, BUILD_CONFIG_PLATFORM_OPTIONS);

    BuildConfigurationKey mapped = mappingValue.map(keyForOptions(modifiedOptions));

    assertThat(keyForOptions(modifiedOptions)).isEqualTo(mapped);
  }

  @Test
  public void testMapNoMappingIfPlatformIsSetAndNoPlatformMapping() throws Exception {
    ImmutableMap<ImmutableSet<String>, Label> flagsToPlatforms =
        ImmutableMap.of(ImmutableSet.of("--cpu=one"), PLATFORM1);

    BuildOptions modifiedOptions = DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.clone();
    modifiedOptions.get(CoreOptions.class).cpu = "one";
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM2);

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(
            ImmutableMap.of(), flagsToPlatforms, BUILD_CONFIG_PLATFORM_OPTIONS);

    BuildConfigurationKey mapped = mappingValue.map(keyForOptions(modifiedOptions));

    assertThat(keyForOptions(modifiedOptions)).isEqualTo(mapped);
  }

  @Test
  public void testDefaultKey() {
    PlatformMappingValue.Key key = PlatformMappingValue.Key.create(null);

    assertThat(key.getWorkspaceRelativeMappingPath())
        .isEqualTo(PlatformOptions.DEFAULT_PLATFORM_MAPPINGS);
    assertThat(key.wasExplicitlySetByUser()).isFalse();
  }

  @Test
  public void testCustomKey() {
    PlatformMappingValue.Key key = PlatformMappingValue.Key.create(PathFragment.create("my/path"));

    assertThat(key.getWorkspaceRelativeMappingPath()).isEqualTo(PathFragment.create("my/path"));
    assertThat(key.wasExplicitlySetByUser()).isTrue();
  }

  private static BuildConfigurationKey keyForOptions(BuildOptions modifiedOptions) {
    return BuildConfigurationKey.withoutPlatformMapping(modifiedOptions);
  }
}
