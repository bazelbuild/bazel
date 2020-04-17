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
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collection;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link PlatformMappingValue}. */
@RunWith(JUnit4.class)
public class PlatformMappingValueTest {

  // We don't actually care about the contents of this set other than that it is passed intact
  // through the mapping logic. The platform fragment in it is purely an example, it could be any
  // set of fragments.
  private static final ImmutableSet<Class<? extends Fragment>> PLATFORM_FRAGMENT_CLASS =
      ImmutableSet.of(PlatformConfiguration.class);

  private static final ImmutableList<Class<? extends FragmentOptions>>
      BUILD_CONFIG_PLATFORM_OPTIONS = ImmutableList.of(CoreOptions.class, PlatformOptions.class);

  private static final Label PLATFORM1 = Label.parseAbsoluteUnchecked("//platforms:one");
  private static final Label PLATFORM2 = Label.parseAbsoluteUnchecked("//platforms:two");

  private static final BuildOptions DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS =
      getDefaultBuildConfigPlatformOptions();
  private static final BuildOptions.OptionsDiffForReconstruction EMPTY_DIFF =
      BuildOptions.diffForReconstruction(
          DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS, DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS);
  private static final Label DEFAULT_TARGET_PLATFORM =
      Label.parseAbsoluteUnchecked("@local_config_platform//:host");

  @Test
  public void testMapNoMappings() throws OptionsParsingException {
    PlatformMappingValue mappingValue =
        new PlatformMappingValue(ImmutableMap.of(), ImmutableMap.of());

    BuildConfigurationValue.Key key =
        BuildConfigurationValue.keyWithoutPlatformMapping(PLATFORM_FRAGMENT_CLASS, EMPTY_DIFF);

    BuildConfigurationValue.Key mapped =
        mappingValue.map(key, DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS);

    assertThat(toMappedOptions(mapped).get(PlatformOptions.class).platforms)
        .containsExactly(DEFAULT_TARGET_PLATFORM);
  }

  @Test
  public void testMapPlatformToFlags() throws Exception {
    ImmutableMap<Label, Collection<String>> platformsToFlags =
        ImmutableMap.of(PLATFORM1, ImmutableList.of("--cpu=one", "--compilation_mode=dbg"));

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(platformsToFlags, ImmutableMap.of());

    BuildOptions modifiedOptions = DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.clone();
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM1);

    BuildConfigurationValue.Key mapped =
        mappingValue.map(keyForOptions(modifiedOptions), DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS);

    assertThat(mapped.getFragments()).isEqualTo(PLATFORM_FRAGMENT_CLASS);

    assertThat(toMappedOptions(mapped).get(CoreOptions.class).cpu).isEqualTo("one");
  }

  @Test
  public void testMapFlagsToPlatform() throws Exception {
    ImmutableMap<Collection<String>, Label> flagsToPlatforms =
        ImmutableMap.of(ImmutableList.of("--cpu=one", "--compilation_mode=dbg"), PLATFORM1);

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(ImmutableMap.of(), flagsToPlatforms);

    BuildOptions modifiedOptions = DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.clone();
    modifiedOptions.get(CoreOptions.class).cpu = "one";
    modifiedOptions.get(CoreOptions.class).compilationMode = CompilationMode.DBG;

    BuildConfigurationValue.Key mapped =
        mappingValue.map(keyForOptions(modifiedOptions), DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS);

    assertThat(mapped.getFragments()).isEqualTo(PLATFORM_FRAGMENT_CLASS);

    assertThat(toMappedOptions(mapped).get(PlatformOptions.class).platforms)
        .containsExactly(PLATFORM1);
  }

  @Test
  public void testMapFlagsToPlatformPriority() throws Exception {
    ImmutableMap<Collection<String>, Label> flagsToPlatforms =
        ImmutableMap.of(
            ImmutableList.of("--cpu=foo", "--compilation_mode=dbg"), PLATFORM1,
            ImmutableList.of("--cpu=foo"), PLATFORM2);

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(ImmutableMap.of(), flagsToPlatforms);

    BuildOptions modifiedOptions = DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.clone();
    modifiedOptions.get(CoreOptions.class).cpu = "foo";

    BuildConfigurationValue.Key mapped =
        mappingValue.map(keyForOptions(modifiedOptions), DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS);

    assertThat(toMappedOptions(mapped).get(PlatformOptions.class).platforms)
        .containsExactly(PLATFORM2);
  }

  @Test
  public void testMapFlagsToPlatformNoneMatching() throws Exception {
    ImmutableMap<Collection<String>, Label> flagsToPlatforms =
        ImmutableMap.of(ImmutableList.of("--cpu=foo", "--compilation_mode=dbg"), PLATFORM1);

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(ImmutableMap.of(), flagsToPlatforms);

    BuildOptions modifiedOptions = DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.clone();
    modifiedOptions.get(CoreOptions.class).cpu = "bar";

    BuildConfigurationValue.Key mapped =
        mappingValue.map(keyForOptions(modifiedOptions), DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS);

    assertThat(toMappedOptions(mapped).get(PlatformOptions.class).platforms)
        .containsExactly(DEFAULT_TARGET_PLATFORM);
  }

  @Test
  public void testMapNoPlatformOptions() throws Exception {
    ImmutableMap<Collection<String>, Label> flagsToPlatforms =
        ImmutableMap.of(ImmutableList.of("--cpu=one"), PLATFORM1);

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(ImmutableMap.of(), flagsToPlatforms);

    BuildOptions options = BuildOptions.of(ImmutableList.of(CoreOptions.class));

    assertThrows(
        IllegalArgumentException.class,
        () -> mappingValue.map(keyForOptions(options), DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS));
  }

  @Test
  public void testMapNoMappingIfPlatformIsSetButNotMatching() throws Exception {
    ImmutableMap<Label, Collection<String>> platformsToFlags =
        ImmutableMap.of(PLATFORM1, ImmutableList.of("--cpu=one", "--compilation_mode=dbg"));
    ImmutableMap<Collection<String>, Label> flagsToPlatforms =
        ImmutableMap.of(ImmutableList.of("--cpu=one"), PLATFORM1);

    BuildOptions modifiedOptions = DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.clone();
    modifiedOptions.get(CoreOptions.class).cpu = "one";
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM2);

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(platformsToFlags, flagsToPlatforms);

    BuildConfigurationValue.Key mapped =
        mappingValue.map(keyForOptions(modifiedOptions), DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS);

    assertThat(keyForOptions(modifiedOptions)).isEqualTo(mapped);
  }

  @Test
  public void testMapNoMappingIfPlatformIsSetAndNoPlatformMapping() throws Exception {
    ImmutableMap<Collection<String>, Label> flagsToPlatforms =
        ImmutableMap.of(ImmutableList.of("--cpu=one"), PLATFORM1);

    BuildOptions modifiedOptions = DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.clone();
    modifiedOptions.get(CoreOptions.class).cpu = "one";
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM2);

    PlatformMappingValue mappingValue =
        new PlatformMappingValue(ImmutableMap.of(), flagsToPlatforms);

    BuildConfigurationValue.Key mapped =
        mappingValue.map(keyForOptions(modifiedOptions), DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS);

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

  private static BuildOptions toMappedOptions(BuildConfigurationValue.Key mapped) {
    return DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS.applyDiff(mapped.getOptionsDiff());
  }

  private static BuildOptions getDefaultBuildConfigPlatformOptions() {
    try {
      return BuildOptions.of(BUILD_CONFIG_PLATFORM_OPTIONS);
    } catch (OptionsParsingException e) {
      throw new RuntimeException(e);
    }
  }

  private static BuildConfigurationValue.Key keyForOptions(BuildOptions modifiedOptions) {
    BuildOptions.OptionsDiffForReconstruction diff =
        BuildOptions.diffForReconstruction(DEFAULT_BUILD_CONFIG_PLATFORM_OPTIONS, modifiedOptions);

    return BuildConfigurationValue.keyWithoutPlatformMapping(PLATFORM_FRAGMENT_CLASS, diff);
  }
}
