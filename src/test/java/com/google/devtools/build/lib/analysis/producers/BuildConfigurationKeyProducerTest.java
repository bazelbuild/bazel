// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.producers;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CommonOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.config.PlatformMappingException;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link BuildConfigurationKeyProducer}. */
@RunWith(JUnit4.class)
public class BuildConfigurationKeyProducerTest extends ProducerTestCase {

  /** Extra options for this test. */
  public static class DummyTestOptions extends FragmentOptions {
    public DummyTestOptions() {}

    @Option(
        name = "option",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "from_default")
    public String option;

    @Option(
        name = "internal_option",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "from_default",
        metadataTags = {OptionMetadataTag.INTERNAL})
    public String internalOption;

    @Option(
        name = "accumulating",
        allowMultiple = true,
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "null")
    public List<String> accumulating;
  }

  /** Test fragment. */
  @RequiresOptions(options = {DummyTestOptions.class})
  public static final class DummyTestOptionsFragment extends Fragment {
    private final BuildOptions buildOptions;

    public DummyTestOptionsFragment(BuildOptions buildOptions) {
      this.buildOptions = buildOptions;
    }

    // Getter required to satisfy AutoCodec.
    public BuildOptions getBuildOptions() {
      return buildOptions;
    }
  }

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(DummyTestOptionsFragment.class);
    return builder.build();
  }

  @Before
  public void writePlatforms() throws Exception {
    scratch.file(
        "platforms/BUILD",
        """
        platform(name = "sample")
        """);
  }

  private void createStarlarkFlag() throws Exception {
    scratch.file(
        "flag/def.bzl",
        """
        def _impl(ctx):
            return []

        basic_flag = rule(
            implementation = _impl,
            build_setting = config.string(flag = True),
        )
        """);

    scratch.file(
        "flag/BUILD",
        """
        load(":def.bzl", "basic_flag")

        basic_flag(
            name = "flag",
            build_setting_default = "from_default",
        )
        """);
  }

  @Test
  public void createKey() throws Exception {
    BuildOptions baseOptions =
        createBuildOptions("--platforms=//platforms:sample", "--internal_option=from_cmd");
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    assertThat(result.getOptions().contains(DummyTestOptions.class)).isTrue();
    assertThat(result.getOptions().get(DummyTestOptions.class).internalOption)
        .isEqualTo("from_cmd");
  }

  @Test
  public void createKey_emptyConfig() throws Exception {
    BuildOptions baseOptions = CommonOptions.EMPTY_OPTIONS;
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    assertThat(result.getOptionsChecksum()).isEqualTo(CommonOptions.EMPTY_OPTIONS.checksum());
  }

  @Test
  public void createKey_platformMapping() throws Exception {
    scratch.file(
        "/workspace/platform_mappings",
        """
        platforms:
          //platforms:sample
            --internal_option=from_mapping_changed
        """);
    invalidatePackages(false);

    BuildOptions baseOptions =
        createBuildOptions("--platforms=//platforms:sample", "--internal_option=from_cmd");
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    assertThat(result.getOptions().contains(DummyTestOptions.class)).isTrue();
    assertThat(result.getOptions().get(DummyTestOptions.class).internalOption)
        .isEqualTo("from_mapping_changed");
  }

  @Test
  public void createKey_platformMapping_invalidFile() throws Exception {
    scratch.file(
        "/workspace/platform_mappings",
        """
        not a mapping file
        """);
    invalidatePackages(false);

    BuildOptions baseOptions = createBuildOptions("--platforms=//platforms:sample");
    // Fails because the mapping file is poorly formed and cannot be parsed.
    assertThrows(PlatformMappingException.class, () -> fetch(baseOptions));
  }

  @Test
  public void createKey_platformMapping_invalidOption() throws Exception {
    scratch.file(
        "/workspace/platform_mappings",
        """
        platforms:
          //platforms:sample
            --fake_option
        """);
    invalidatePackages(false);

    BuildOptions baseOptions = createBuildOptions("--platforms=//platforms:sample");
    // Fails because the changed platform has an invalid mapping.
    var e = assertThrows(PlatformMappingException.class, () -> fetch(baseOptions));
    assertThat(e).hasMessageThat().contains("Unrecognized option: --fake_option");
  }

  @Test
  public void createKey_platformFlags_native() throws Exception {
    scratch.overwriteFile(
        "platforms/BUILD",
        """
        platform(
            name = "sample",
            flags = [
                "--internal_option=from_platform",
            ],
        )
        """);
    invalidatePackages(false);

    BuildOptions baseOptions = createBuildOptions("--platforms=//platforms:sample");
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    assertThat(result.getOptions().contains(DummyTestOptions.class)).isTrue();
    assertThat(result.getOptions().get(DummyTestOptions.class).internalOption)
        .isEqualTo("from_platform");
  }

  @Test
  public void createKey_platformFlags_starlark() throws Exception {
    createStarlarkFlag();
    scratch.overwriteFile(
        "platforms/BUILD",
        """
        platform(
            name = "sample",
            flags = [
                "--//flag=from_platform",
            ],
        )
        """);
    invalidatePackages(false);

    BuildOptions baseOptions = createBuildOptions("--platforms=//platforms:sample");
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    assertThat(result.getOptions().getStarlarkOptions())
        .containsAtLeast(Label.parseCanonicalUnchecked("//flag"), "from_platform");
  }

  @Test
  public void createKey_platformFlags_override_native() throws Exception {
    scratch.overwriteFile(
        "platforms/BUILD",
        """
        platform(
            name = "sample",
            flags = [
                "--option=from_platform",
            ],
        )
        """);
    invalidatePackages(false);

    BuildOptions baseOptions =
        createBuildOptions("--platforms=//platforms:sample", "--option=from_cli");
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    assertThat(result.getOptions().contains(DummyTestOptions.class)).isTrue();
    assertThat(result.getOptions().get(DummyTestOptions.class).option).isEqualTo("from_platform");
  }

  @Test
  public void createKey_platformFlags_override_starlark() throws Exception {
    createStarlarkFlag();
    scratch.overwriteFile(
        "platforms/BUILD",
        """
        platform(
            name = "sample",
            flags = [
                "--//flag=from_platform",
            ],
        )
        """);
    invalidatePackages(false);

    BuildOptions baseOptions =
        createBuildOptions("--platforms=//platforms:sample", "--//flag=from_cli");
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    assertThat(result.getOptions().getStarlarkOptions())
        .containsAtLeast(Label.parseCanonicalUnchecked("//flag"), "from_platform");
  }

  @Test
  public void createKey_platformFlags_resetToDefault_native() throws Exception {
    scratch.overwriteFile(
        "platforms/BUILD",
        """
        platform(
            name = "sample",
            flags = [
                "--option=from_default",
            ],
        )
        """);
    invalidatePackages(false);

    BuildOptions baseOptions =
        createBuildOptions("--platforms=//platforms:sample", "--option=from_cli");
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    assertThat(result.getOptions().contains(DummyTestOptions.class)).isTrue();
    assertThat(result.getOptions().get(DummyTestOptions.class).option).isEqualTo("from_default");
  }

  // Regression test for https://github.com/bazelbuild/bazel/issues/23147
  @Test
  public void createKey_platformFlags_resetToDefault_starlark() throws Exception {
    createStarlarkFlag();
    scratch.overwriteFile(
        "platforms/BUILD",
        """
        platform(
            name = "sample",
            flags = [
                "--//flag=from_default",
            ],
        )
        """);
    invalidatePackages(false);

    BuildOptions baseOptions =
        createBuildOptions("--platforms=//platforms:sample", "--//flag=from_cli");
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    // Default key values should not be present in starlark options.
    assertThat(result.getOptions().getStarlarkOptions())
        .doesNotContainKey(Label.parseCanonicalUnchecked("//flag"));
  }

  @Test
  // Re-enable this once merging repeatable flags works properly. Also add a corresponding Starlark
  // flag to test.
  @Ignore("https://github.com/bazelbuild/bazel/issues/22453")
  public void createKey_platformFlags_accumulate() throws Exception {
    scratch.overwriteFile(
        "platforms/BUILD",
        """
        platform(
            name = "sample",
            flags = [
                "--accumulating=from_platform",
            ],
        )
        """);
    invalidatePackages(false);

    BuildOptions baseOptions =
        createBuildOptions("--platforms=//platforms:sample", "--accumulating=from_cli");
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    assertThat(result.getOptions().contains(DummyTestOptions.class)).isTrue();
    assertThat(result.getOptions().get(DummyTestOptions.class).accumulating)
        .containsExactly("from_cli", "from_platform")
        .inOrder();
  }

  @Test
  public void createKey_platformFlags_invalidPlatform() throws Exception {
    scratch.overwriteFile(
        "platforms/BUILD",
        """
        filegroup(name = "sample")
        """);
    invalidatePackages(false);

    BuildOptions baseOptions = createBuildOptions("--platforms=//platforms:sample");
    assertThrows(InvalidPlatformException.class, () -> fetch(baseOptions));
  }

  @Test
  public void createKey_platformFlags_invalidOption() throws Exception {
    scratch.overwriteFile(
        "platforms/BUILD",
        """
        platform(
            name = "sample",
            flags = [
                "--fake_option_doesnt_exist=from_platform",
            ],
        )
        """);
    invalidatePackages(false);

    BuildOptions baseOptions = createBuildOptions("--platforms=//platforms:sample");
    assertThrows(OptionsParsingException.class, () -> fetch(baseOptions));
  }

  @Test
  public void createKey_platformFlags_overridesMapping() throws Exception {
    scratch.file(
        "/workspace/platform_mappings",
        """
        platforms:
          //platforms:sample
            --internal_option=from_mapping
        """);
    scratch.overwriteFile(
        "platforms/BUILD",
        """
        platform(
            name = "sample",
            flags = [
                "--internal_option=from_platform",
            ],
        )
        """);
    invalidatePackages(false);

    BuildOptions baseOptions = createBuildOptions("--platforms=//platforms:sample");
    BuildConfigurationKey result = fetch(baseOptions);

    assertThat(result).isNotNull();
    assertThat(result.getOptions().contains(DummyTestOptions.class)).isTrue();
    assertThat(result.getOptions().get(DummyTestOptions.class).internalOption)
        .isEqualTo("from_platform");
  }

  private static final String CONTEXT = "context";

  private BuildConfigurationKey fetch(BuildOptions options)
      throws InterruptedException,
          OptionsParsingException,
          PlatformMappingException,
          InvalidPlatformException {
    Sink sink = new Sink();
    BuildConfigurationKeyProducer<String> producer =
        new BuildConfigurationKeyProducer<>(
            sink, StateMachine.DONE, new BuildConfigurationKeyCache(), CONTEXT, options);
    // Ignore the return value: sink will either return a result or re-throw whatever exception it
    // received from the producer.
    var unused = executeProducer(producer);
    return sink.options(CONTEXT);
  }

  /** Receiver for platform info from {@link PlatformProducer}. */
  private static class Sink implements BuildConfigurationKeyProducer.ResultSink<String> {
    @Nullable private OptionsParsingException optionsParsingException;
    @Nullable private PlatformMappingException platformMappingException;
    @Nullable private InvalidPlatformException invalidPlatformException;
    @Nullable private String context;
    @Nullable private BuildConfigurationKey key;

    @Override
    public void acceptOptionsParsingError(OptionsParsingException e) {
      this.optionsParsingException = e;
    }

    @Override
    public void acceptPlatformMappingError(PlatformMappingException e) {
      this.platformMappingException = e;
    }

    @Override
    public void acceptPlatformFlagsError(InvalidPlatformException e) {
      this.invalidPlatformException = e;
    }

    @Override
    public void acceptTransitionedConfiguration(String context, BuildConfigurationKey key) {
      this.context = context;
      this.key = key;
    }

    BuildConfigurationKey options(String expectedContext)
        throws OptionsParsingException, PlatformMappingException, InvalidPlatformException {
      if (this.optionsParsingException != null) {
        throw this.optionsParsingException;
      }
      if (this.platformMappingException != null) {
        throw this.platformMappingException;
      }
      if (this.invalidPlatformException != null) {
        throw this.invalidPlatformException;
      }
      if (this.key != null) {
        assertThat(this.context).isEqualTo(expectedContext);
        return this.key;
      }
      throw new IllegalStateException("Value and exception not set");
    }
  }
}
