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
package com.google.devtools.build.lib.skyframe.config;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.skyframe.EvaluationResult;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FlagSetFunction}. */
@RunWith(JUnit4.class)
public final class FlagSetsFunctionTest extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(DummyTestFragment.class);
    return builder.build();
  }

  /**
   * Given "//foo:myflag" and "default_value", creates the BUILD and .bzl files to realize a
   * string_flag with that label and default value.
   */
  private void createStringFlag(String labelName, String defaultValue) throws Exception {
    String flagDir = labelName.substring(2, labelName.indexOf(":"));
    String flagName = labelName.substring(labelName.indexOf(":") + 1);
    scratch.file(
        flagDir + "/build_settings.bzl",
        """
string_flag = rule(implementation = lambda ctx: [], build_setting = config.string(flag = True))
""");
    scratch.file(
        flagDir + "/BUILD",
        """
        load(":build_settings.bzl", "string_flag")
        string_flag(
            name = "%s",
            build_setting_default = "%s",
        )
        """
            .formatted(flagName, defaultValue));
  }

  @Test
  public void flagSetsFunction_returns_modified_buildOptions() throws Exception {
    rewriteWorkspace("workspace(name = 'my_workspace')");
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--platforms=//buildenv/platforms/android:x86'],
        }
        """);
    scratch.file("test/BUILD");
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    // given original BuildOptions and a valid key
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ false);
    FlagSetValue flagSetsValue = executeFunction(key);

    // expects the modified BuildOptions
    assertThat(flagSetsValue.getTopLevelBuildOptions().get(PlatformOptions.class).platforms)
        .containsExactly(Label.parseCanonical("//buildenv/platforms/android:x86"));
  }

  @Test
  public void given_unknown_sclConfig_flagSetsFunction_returns_original_buildOptions()
      throws Exception {
    rewriteWorkspace("workspace(name = 'my_workspace')");
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--platforms=//buildenv/platforms/android:x86'],
        }
        """);
    scratch.file("test/BUILD");
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    // given valid project file but a nonexistent scl config
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            "unknown_config",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ false);
    FlagSetValue flagSetsValue = executeFunction(key);

    // expects the original BuildOptions
    assertThat(flagSetsValue.getTopLevelBuildOptions()).isEqualTo(buildOptions);
  }

  @Test
  public void flagSetsFunction_returns_original_buildOptions() throws Exception {
    // given original BuildOptions and an empty scl config name
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            "",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ false);
    FlagSetValue flagSetsValue = executeFunction(key);

    // expects the original BuildOptions
    assertThat(flagSetsValue.getTopLevelBuildOptions()).isEqualTo(buildOptions);
  }

  @Test
  public void noEnforceCanonicalConfigs_unknownConfigisNoop() throws Exception {
    scratch.file("test/PROJECT.scl", "");
    scratch.file("test/BUILD");
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            "random_config_name",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ false);
    FlagSetValue flagSetsValue = executeFunction(key);

    // Without enforced configs, unknown configs are a no-op.
    assertThat(flagSetsValue.getTopLevelBuildOptions()).isEqualTo(buildOptions);
  }

  @Test
  public void enforceCanonicalConfigsSupportedConfig() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "other_config": ['--//test:myflag=other_config_value'],
        }
        supported_configs = {
          "test_config": "User documentation for what this config means",
        }
        default_config = "test_config"
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ true);
    FlagSetValue flagSetsValue = executeFunction(key);

    assertThat(
            flagSetsValue
                .getTopLevelBuildOptions()
                .getStarlarkOptions()
                .get(Label.parseCanonical("//test:myflag")))
        .isEqualTo("test_config_value");
  }

  @Test
  public void enforceCanonicalConfigsExtraNativeFlag_withSclConfig_fails() throws Exception {
    scratch.file(
        "test/build_settings.bzl",
        """
string_flag = rule(implementation = lambda ctx: [], build_setting = config.string(flag = True))
""");
    scratch.file(
        "test/BUILD",
        """
        load("//test:build_settings.bzl", "string_flag")
        string_flag(
            name = "myflag",
            build_setting_default = "default",
        )
        """);
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "other_config": ['--//test:myflag=other_config_value'],
        }
        supported_configs = {
          "test_config": "User documentation for what this config means",
        }
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions = createBuildOptions("--define=foo=bar");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableSet.of("--define=foo=bar"),
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown).hasMessageThat().contains("Found [--define=foo=bar]");
  }

  @Test
  public void enforceCanonicalConfigsExtraNativeFlag_noSupportedConfigs_passes() throws Exception {
    scratch.file(
        "test/build_settings.bzl",
        """
string_flag = rule(implementation = lambda ctx: [], build_setting = config.string(flag = True))
""");
    scratch.file(
        "test/BUILD",
        """
        load("//test:build_settings.bzl", "string_flag")
        string_flag(
            name = "myflag",
            build_setting_default = "default",
        )
        """);
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "other_config": ['--//test:myflag=other_config_value'],
        }
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions = createBuildOptions("--define=foo=bar");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            null,
            buildOptions,
            /* userOptions= */ ImmutableSet.of("--define=foo=bar"),
            /* enforceCanonical= */ true);

    var unused = executeFunction(key);
    assertDoesNotContainEvent("--scl_config must be the only configuration-affecting flag");
  }

  @Test
  public void enforceCanonicalConfigsExtraStarlarkFlag_fails() throws Exception {
    scratch.file(
        "test/build_settings.bzl",
        """
string_flag = rule(implementation = lambda ctx: [], build_setting = config.string(flag = True))
""");
    scratch.file(
        "test/BUILD",
        """
        load("//test:build_settings.bzl", "string_flag")
        string_flag(
            name = "myflag",
            build_setting_default = "default",
        )
        string_flag(
            name = "starlark_flags_always_affect_configuration",
            build_setting_default = "default",
        )
        """);
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "other_config": ['--//test:myflag=other_config_value'],
        }
        supported_configs = {
          "test_config": "User documentation for what this config means",
        }
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions =
        createBuildOptions("--//test:starlark_flags_always_affect_configuration=yes_they_do");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(
                "--//test:starlark_flags_always_affect_configuration=yes_they_do"),
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains("Found [--//test:starlark_flags_always_affect_configuration=yes_they_do]");
  }

  @Test
  public void noEnforceCanonicalConfigsExtraFlag_passes() throws Exception {
    scratch.file(
        "test/build_settings.bzl",
        """
string_flag = rule(implementation = lambda ctx: [], build_setting = config.string(flag = True))
""");
    scratch.file(
        "test/BUILD",
        """
        load("//test:build_settings.bzl", "string_flag")
        string_flag(
            name = "myflag",
            build_setting_default = "default",
        )
        """);
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "other_config": ['--//test:myflag=other_config_value'],
        }
        supported_configs = {
          "test_config": "User documentation for what this config means",
        }
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions = createBuildOptions("--define=foo=bar");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableSet.of("--define=foo=bar"),
            /* enforceCanonical= */ false);

    var unused = executeFunction(key);
    assertDoesNotContainEvent("--scl_config must be the only configuration-affecting flag");
  }

  @Test
  public void enforceCanonicalConfigsUnsupportedConfig() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "other_config": ['--//test:myflag=other_config_value'],
        }
        supported_configs = {
          "test_config": "User documentation for what this config means",
        }
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            "other_config",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains("--scl_config=other_config is not a valid configuration for this project");
  }

  @Test
  public void enforceCanonicalConfigsNonExistentConfig() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "other_config": ['--//test:myflag=other_config_value'],
        }
        supported_configs = {
          "test_config": "User documentation for what this config means",
        }
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            "non_existent_config",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains("--scl_config=non_existent_config is not a valid configuration for this project");
  }

  @Test
  public void enforceCanonicalConfigsNoSclConfigFlagNoDefaultConfig() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "other_config": ['--//test:myflag=other_config_value'],
        }
        supported_configs = {
          "test_config": "User documentation for what this config means",
        }
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains(
            "This project's builds must set --scl_config because no default_config is defined");
  }

  @Test
  public void enforceCanonicalConfigsNoSclConfigFlagUnsupportedDefaultConfig() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "unsupported_config": ['--//test:myflag=other_config_value'],
        }
        supported_configs = {
          "test_config": "User documentation for what this config means",
        }
        default_config = "unsupported_config"
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains(
            "This project's builds must set --scl_config because default_config does not refer to a"
                + " supported config: unsupported_config");
  }

  @Test
  public void enforceCanonicalConfigsNoSclConfigFlagNonexistentDefaultConfig() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "other_config": ['--//test:myflag=other_config_value'],
        }
        supported_configs = {
          "test_config": "User documentation for what this config means",
        }
        default_config = "nonexistent_config"
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains(
            "This project's builds must set --scl_config because default_config refers to a"
                + " nonexistent config: nonexistent_config");
  }

  @Test
  public void enforceCanonicalConfigsNoSclConfigFlagValidDefaultConfig() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "other_config": ['--//test:myflag=other_config_value'],
        }
        supported_configs = {
          "test_config": "User documentation for what this config means",
        }
        default_config = "test_config"
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ true);
    FlagSetValue flagSetsValue = executeFunction(key);

    assertThat(
            flagSetsValue
                .getTopLevelBuildOptions()
                .getStarlarkOptions()
                .get(Label.parseCanonical("//test:myflag")))
        .isEqualTo("test_config_value");

    assertContainsEventWithFrequency("Applying flags from the default config", 1);
  }

  @Test
  public void enforceCanonicalConfigsNoSupportedConfigsWithNoSclConfig() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "other_config": ['--//test:myflag=other_config_value'],
        }
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ true);
    FlagSetValue flagSetsValue = executeFunction(key);

    assertThat(flagSetsValue.getTopLevelBuildOptions().getStarlarkOptions()).isEmpty();
  }

  @Test
  public void enforceCanonicalConfigsNoSupportedConfigsWithSclConfig() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "test_config": ['--//test:myflag=test_config_value'],
          "other_config": ['--//test:myflag=other_config_value'],
        }
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "test_config",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ true);
    FlagSetValue flagSetsValue = executeFunction(key);

    assertThat(
            flagSetsValue
                .getTopLevelBuildOptions()
                .getStarlarkOptions()
                .get(Label.parseCanonical("//test:myflag")))
        .isEqualTo("test_config_value");
  }

  @Test
  public void clearUserDocumentationOfSupportedConfigs() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        configs = {
          "debug": ['--//test:myflag=debug_value'],
          "release": ['--//test:myflag=debug_value'],
        }
        supported_configs = {
          "debug": "build binaries for local debugging",
          "release": "build binaries for product releases",
        }
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "",
            buildOptions,
            /* userOptions= */ ImmutableSet.of(),
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains(
            """
This project's builds must set --scl_config because no default_config is defined.

This project supports:
  --scl_config=debug: build binaries for local debugging
  --scl_config=release: build binaries for product releases

This policy is defined in test/PROJECT.scl.""");
  }

  private FlagSetValue executeFunction(FlagSetValue.Key key) throws Exception {
    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    EvaluationResult<FlagSetValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, /* keepGoing= */ false, reporter);
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE, Optional.empty())));
    if (result.hasError()) {
      throw result.getError(key).getException();
    }
    return result.get(key);
  }
}
