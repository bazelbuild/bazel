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
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
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
