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
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Optional;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link PlatformMappingFunction}.
 *
 * <p>Note that all parsing tests are located in {@link PlatformMappingFunctionParserTest}.
 */
@RunWith(JUnit4.class)
public final class PlatformMappingFunctionTest extends BuildViewTestCase {

  private static final Label PLATFORM1 = Label.parseCanonicalUnchecked("//platforms:one");

  private static final Label DEFAULT_TARGET_PLATFORM =
      Label.parseCanonicalUnchecked("@local_config_platform//:host");

  private BuildOptions defaultBuildOptions;

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(DummyTestFragment.class);
    return builder.build();
  }

  @Before
  public void setDefaultBuildOptions() {
    defaultBuildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
  }

  @Test
  public void testMappingFileDoesNotExist() {
    MissingInputFileException exception =
        assertThrows(
            MissingInputFileException.class,
            () ->
                executeFunction(
                    PlatformMappingValue.Key.create(PathFragment.create("random_location"))));
    assertThat(exception).hasMessageThat().contains("random_location");
  }

  @Test
  public void testMappingFileDoesNotExistDefaultLocation() throws Exception {
    PlatformMappingValue platformMappingValue =
        executeFunction(PlatformMappingValue.Key.create(null));

    BuildConfigurationKey key = BuildConfigurationKey.withoutPlatformMapping(defaultBuildOptions);

    BuildConfigurationKey mapped = platformMappingValue.map(key);

    assertThat(mapped.getOptions().get(PlatformOptions.class).platforms)
        .containsExactly(DEFAULT_TARGET_PLATFORM);
  }

  @Test
  public void testMappingFileIsDirectory() throws Exception {
    scratch.dir("somedir");

    MissingInputFileException exception =
        assertThrows(
            MissingInputFileException.class,
            () -> executeFunction(PlatformMappingValue.Key.create(PathFragment.create("somedir"))));
    assertThat(exception).hasMessageThat().contains("somedir");
  }

  @Test
  public void testMappingFileIsRead() throws Exception {
    scratch.file(
        "my_mapping_file",
        "platforms:", // Force line break
        "  //platforms:one", // Force line break
        "    --cpu=one");

    PlatformMappingValue platformMappingValue =
        executeFunction(PlatformMappingValue.Key.create(PathFragment.create("my_mapping_file")));

    BuildOptions modifiedOptions = defaultBuildOptions.clone();
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM1);

    BuildConfigurationKey mapped = platformMappingValue.map(keyForOptions(modifiedOptions));

    assertThat(mapped.getOptions().get(CoreOptions.class).cpu).isEqualTo("one");
  }

  @Test
  public void testMappingFileIsRead_fromAlternatePackagePath() throws Exception {
    scratch.setWorkingDir("/other/package/path");
    scratch.file("WORKSPACE");
    setPackageOptions("--package_path=/other/package/path");
    scratch.file(
        "my_mapping_file",
        "platforms:", // Force line break
        "  //platforms:one", // Force line break
        "    --cpu=one");

    PlatformMappingValue platformMappingValue =
        executeFunction(PlatformMappingValue.Key.create(PathFragment.create("my_mapping_file")));

    BuildOptions modifiedOptions = defaultBuildOptions.clone();
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM1);

    BuildConfigurationKey mapped = platformMappingValue.map(keyForOptions(modifiedOptions));

    assertThat(mapped.getOptions().get(CoreOptions.class).cpu).isEqualTo("one");
  }

  @Test
  public void handlesNoWorkspaceFile() throws Exception {
    scratch.setWorkingDir("/other/package/path");
    scratch.file(
        "my_mapping_file",
        "platforms:", // Force line break
        "  //platforms:one", // Force line break
        "    --cpu=one");
    setPackageOptions("--package_path=/other/package/path");

    PlatformMappingValue platformMappingValue =
        executeFunction(PlatformMappingValue.Key.create(PathFragment.create("my_mapping_file")));
    BuildOptions modifiedOptions = defaultBuildOptions.clone();
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM1);

    BuildConfigurationKey mapped = platformMappingValue.map(keyForOptions(modifiedOptions));

    assertThat(mapped.getOptions().get(CoreOptions.class).cpu).isEqualTo("one");
  }

  @Test
  public void multiplePackagePaths() throws Exception {
    scratch.setWorkingDir("/other/package/path");
    scratch.file(
        "my_mapping_file",
        "platforms:", // Force line break
        "  //platforms:one", // Force line break
        "    --cpu=one");
    setPackageOptions("--package_path=%workspace%:/other/package/path");

    PlatformMappingValue platformMappingValue =
        executeFunction(PlatformMappingValue.Key.create(PathFragment.create("my_mapping_file")));

    BuildOptions modifiedOptions = defaultBuildOptions.clone();
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM1);

    BuildConfigurationKey mapped = platformMappingValue.map(keyForOptions(modifiedOptions));

    assertThat(mapped.getOptions().get(CoreOptions.class).cpu).isEqualTo("one");
  }

  @Test
  public void multiplePackagePathsFirstWins() throws Exception {
    scratch.file(
        "my_mapping_file",
        "platforms:", // Force line break
        "  //platforms:one", // Force line break
        "    --cpu=one");
    scratch.setWorkingDir("/other/package/path");
    scratch.file(
        "my_mapping_file",
        "platforms:", // Force line break
        "  //platforms:one", // Force line break
        "    --cpu=two");
    setPackageOptions("--package_path=%workspace%:/other/package/path");

    PlatformMappingValue platformMappingValue =
        executeFunction(PlatformMappingValue.Key.create(PathFragment.create("my_mapping_file")));

    BuildOptions modifiedOptions = defaultBuildOptions.clone();
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM1);

    BuildConfigurationKey mapped = platformMappingValue.map(keyForOptions(modifiedOptions));

    assertThat(mapped.getOptions().get(CoreOptions.class).cpu).isEqualTo("one");
  }

  // Internal flags (OptionMetadataTag.INTERNAL) cannot be set from the command-line, but
  // platform mapping needs to access them.
  @Test
  public void ableToChangeInternalOption() throws Exception {
    scratch.file(
        "my_mapping_file",
        "platforms:", // Force line break
        "  //platforms:one", // Force line break
        "    --internal foo=something_new");

    PlatformMappingValue platformMappingValue =
        executeFunction(PlatformMappingValue.Key.create(PathFragment.create("my_mapping_file")));

    BuildOptions modifiedOptions = defaultBuildOptions.clone();
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM1);

    BuildConfigurationKey mapped = platformMappingValue.map(keyForOptions(modifiedOptions));

    assertThat(mapped.getOptions().get(DummyTestFragment.DummyTestOptions.class).internalFoo)
        .isEqualTo("something_new");
  }

  @Test
  public void starlarkFlagMapping() throws Exception {
    scratch.file(
        "test/build_setting.bzl",
        "def _impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _impl,",
        "  build_setting = config.string(flag=True)",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:build_setting.bzl', 'string_flag')",
        "string_flag(name = 'my_string_flag', build_setting_default = 'default value')");
    scratch.file(
        "my_mapping_file",
        "platforms:", // Force line break
        "  //platforms:one", // Force line break
        "    --//test:my_string_flag=mapped_value");

    PlatformMappingValue platformMappingValue =
        executeFunction(PlatformMappingValue.Key.create(PathFragment.create("my_mapping_file")));

    BuildOptions modifiedOptions = defaultBuildOptions.clone();
    modifiedOptions.get(PlatformOptions.class).platforms = ImmutableList.of(PLATFORM1);
    BuildConfigurationKey mapped = platformMappingValue.map(keyForOptions(modifiedOptions));

    assertThat(mapped.getOptions().getStarlarkOptions())
        .containsExactly(Label.parseCanonical("//test:my_string_flag"), "mapped_value");
  }

  @Test
  public void badStarlarkFlag() throws Exception {
    scratch.file("test/BUILD"); // Set up a valid package but invalid flag.
    scratch.file(
        "my_mapping_file",
        "platforms:", // Force line break
        "  //platforms:one", // Force line break
        "    --//test:this_flag_doesnt_exist=mapped_value");

    assertThrows(
        "Failed to load //test:this_flag_doesnt_exist",
        OptionsParsingException.class,
        () ->
            executeFunction(
                PlatformMappingValue.Key.create(PathFragment.create("my_mapping_file"))));
  }

  @Test
  public void platformTransitionWithStarlarkFlagMapping() throws Exception {
    // Define a Starlark flag:
    scratch.file(
        "test/flags/build_setting.bzl",
        "def _impl(ctx):",
        "  return []",
        "string_flag = rule(",
        "  implementation = _impl,",
        "  build_setting = config.string(flag=True)",
        ")");
    scratch.file(
        "test/flags/BUILD",
        "load('//test/flags:build_setting.bzl', 'string_flag')",
        "string_flag(name = 'my_string_flag', build_setting_default = 'default value')");

    // Define a custom platform and mapping from that platform to the flag:
    scratch.file("test/platforms/BUILD", "platform(", "    name = 'my_platform',", ")");
    scratch.file(
        "my_mapping_file",
        "platforms:", // Force line break
        "  //test/platforms:my_platform", // Force line break
        "    --//test/flags:my_string_flag=platform-mapped value");

    // Define a rule that platform-transitions its deps:
    scratch.overwriteFile(
        "tools/allowlists/function_transition_allowlist/BUILD",
        "package_group(",
        "    name = 'function_transition_allowlist',",
        "    packages = [",
        "        '//test/...',",
        "    ],",
        ")");
    scratch.file(
        "test/starlark/rules.bzl",
        "def transition_func(settings, attr):",
        "  return {'//command_line_option:platforms': '//test/platforms:my_platform'}",
        "my_transition = transition(",
        "  implementation = transition_func,",
        "  inputs = [],",
        "  outputs = ['//command_line_option:platforms']",
        ")",
        "transition_rule = rule(",
        "  implementation = lambda ctx: [],",
        "  attrs = {",
        "    'dep':  attr.label(cfg = my_transition),",
        "    '_allowlist_function_transition': attr.label(",
        "        default = '//tools/allowlists/function_transition_allowlist',",
        "    ),",
        "  }",
        ")");
    scratch.file("test/starlark/BUILD");

    // Define a target to build and its dep:
    scratch.file(
        "test/BUILD",
        "load('//test/starlark:rules.bzl', 'transition_rule')",
        "transition_rule(name = 'main', dep = ':dep')",
        "transition_rule(name = 'dep')");

    // Set the Starlark flag explicitly. Otherwise it won't show up at all in the top-level config's
    // getOptions().getStarlarkOptions() map.
    useConfiguration(
        /* starlarkOptions= */ ImmutableMap.of("//test/flags:my_string_flag", "top-level value"),
        /* args...= */ "--platform_mappings=my_mapping_file");
    ConfiguredTarget main = getConfiguredTarget("//test:main");
    ConfiguredTarget dep = getDirectPrerequisite(main, "//test:dep");

    assertThat(
            getConfiguration(main)
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseCanonical("//test/flags:my_string_flag")))
        .isEqualTo("top-level value");
    assertThat(
            getConfiguration(dep)
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseCanonical("//test/flags:my_string_flag")))
        .isEqualTo("platform-mapped value");
  }

  private PlatformMappingValue executeFunction(PlatformMappingValue.Key key) throws Exception {
    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE, Optional.empty())));
    EvaluationResult<PlatformMappingValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, /*keepGoing=*/ false, reporter);
    if (result.hasError()) {
      throw result.getError(key).getException();
    }
    return result.get(key);
  }

  private static BuildConfigurationKey keyForOptions(BuildOptions modifiedOptions) {
    return BuildConfigurationKey.withoutPlatformMapping(modifiedOptions);
  }
}
