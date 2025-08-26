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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.runtime.ConfigFlagDefinitions;
import com.google.devtools.build.lib.skyframe.ProjectValue.BuildableUnit;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import com.google.testing.junit.testparameterinjector.TestParameters.TestParametersValues;
import com.google.testing.junit.testparameterinjector.TestParametersValuesProvider;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link FlagSetFunction}. */
@RunWith(TestParameterInjector.class)
public final class FlagSetFunctionTest extends BuildViewTestCase {
  // TODO: b/409377907 - Most of this enforcement has been moved to ProjectFunction. Move the
  // corresponding tests to ProjectFunctionTest.

  @Before
  public void setUp() throws Exception {
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    writeProjectSclDefinition("test/project_proto.scl");
  }

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(DummyTestFragment.class);
    return builder.build();
  }

  /**
   * Asserts a {@link FlagSetValue} contains a given kind of event with a given message that
   * occurred a given number of times.
   *
   * <p>Only applies to messages that are expected to persistently display, even on Skyframe cache
   * hits: see {@link FlagSetValue#getPersistentMessages}.
   */
  private void assertContainsPersistentMessage(
      FlagSetValue value, EventKind kind, int frequency, String message) {
    int count = 0;
    for (Event event : value.getPersistentMessages()) {
      if (event.getKind() != kind) {
        continue;
      }
      count++;
      assertThat(event.getMessage()).contains(message);
    }
    assertThat(count).isEqualTo(frequency);
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
        string_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.string(flag = True),
            attrs = {
                "scope": attr.string(default = "universal"),
            },
        )
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
    scratch.file(
        "test/PROJECT.scl",
        """
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--platforms=//buildenv/platforms/android:x86"]
              ),
          ],
        )
        """);
    scratch.file("test/BUILD");
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    // given original BuildOptions and a valid key
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);
    FlagSetValue flagSetsValue = executeFunction(key);
    // expects the modified BuildOptions
    assertThat(flagSetsValue.getOptionsFromFlagset())
        .containsExactly("--platforms=//buildenv/platforms/android:x86");
  }

  @Test
  public void given_unknown_sclConfig_flagSetsFunction_returns_original_buildOptions()
      throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--platforms=//buildenv/platforms/android:x86"]
              ),
          ],
        )
        """);
    scratch.file("test/BUILD");
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    // given valid project file but a nonexistent scl config
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "unknown_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ false);
    FlagSetValue flagSetsValue = executeFunction(key);

    // expects the original BuildOptions
    assertThat(flagSetsValue.getOptionsFromFlagset()).isEmpty();
  }

  @Test
  public void flagSetsFunction_returns_original_buildOptions() throws Exception {
    // given original BuildOptions and an empty scl config name
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ false);
    FlagSetValue flagSetsValue = executeFunction(key);

    // expects the original BuildOptions
    assertThat(flagSetsValue.getOptionsFromFlagset()).isEmpty();
  }

  @Test
  public void invalidEnforcementPolicy_fails() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          enforcement_policy = "invalid",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--platforms=//buildenv/platforms/android:x86"]
              ),
          ],
        )
        """);
    scratch.file("test/BUILD");
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    // given original BuildOptions and a valid key
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);
    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains("invalid enforcement_policy 'invalid' in //test:PROJECT.scl");
  }

  @Test
  public void noEnforceCanonicalConfigs_noConfigsIsNoop() throws Exception {
    scratch.file("test/PROJECT.scl", "");
    scratch.file("test/BUILD");
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "random_config_name",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ false);
    FlagSetValue flagSetsValue = executeFunction(key);

    // Without enforced configs, unknown configs are a no-op.
    assertThat(flagSetsValue.getOptionsFromFlagset()).isEmpty();
    assertContainsEvent("Ignoring --scl_config=random_config_name");
  }

  @Test
  public void noEnforceCanonicalConfigs_sclConfigWarns() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--define=bar=bar"]
              ),
          ],
        )
        """);
    scratch.file("test/BUILD");
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ false);

    FlagSetValue flagSetsValue = executeFunction(key);

    assertThat(flagSetsValue.getOptionsFromFlagset()).isEmpty();
    assertContainsEvent(
        "Ignoring --scl_config=test_config because --enforce_project_configs is not set");
  }

  @Test
  public void enforceCanonicalConfigs_noConfigsIsNoop() throws Exception {
    scratch.file("test/PROJECT.scl", "");
    scratch.file("test/BUILD");
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "fake_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ false);
    FlagSetValue flagSetsValue = executeFunction(key);

    assertThat(flagSetsValue.getOptionsFromFlagset()).isEmpty();
  }

  @Test
  public void enforceCanonicalConfigsValidConfig() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value"],
                  is_default = True,
              ),
              buildable_unit_pb2.BuildableUnit.create(
                  name = "other_config",
                  flags = ["--//test:myflag=other_config_value"]
              ),
          ],
        )
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "other_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);
    FlagSetValue flagSetsValue = executeFunction(key);

    assertThat(flagSetsValue.getOptionsFromFlagset())
        .contains("--//test:myflag=other_config_value");
    assertContainsPersistentMessage(
        flagSetsValue,
        EventKind.INFO,
        /* frequency= */ 1,
        "Applying flags from the config 'other_config'");
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
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          enforcement_policy = "strict",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value"]
              ),
              buildable_unit_pb2.BuildableUnit.create(
                  name = "other_config",
                  flags = ["--//test:myflag=other_config_value"]
              ),
          ],
        )
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions = createBuildOptions("--define=foo=bar");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of("--define=foo=bar", ""),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown).hasMessageThat().contains("Found ['--define=foo=bar']");
  }

  @Test
  public void enforceCanonicalConfigsFlag_warnPolicy_passes() throws Exception {
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
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          enforcement_policy = "warn",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value"]
              ),
              buildable_unit_pb2.BuildableUnit.create(
                  name = "other_config",
                  flags = ["--//test:myflag=other_config_value"]
              ),
          ],
        )
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions = createBuildOptions("--define=foo=bar");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of("--define=foo=bar", ""),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    assertContainsPersistentMessage(
        executeFunction(key),
        EventKind.WARNING,
        /* frequency= */ 1,
        "also sets output-affecting flags in the command line or user bazelrc:"
            + " ['--define=foo=bar']");
  }

  @Test
  public void enforceCanonicalConfigsFlag_compatiblePolicy_unrelatedFlag_warns() throws Exception {
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
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          enforcement_policy = "compatible",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value"]
              ),
              buildable_unit_pb2.BuildableUnit.create(
                  name = "other_config",
                  flags = ["--//test:myflag=other_config_value"]
              ),
          ],
        )
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions = createBuildOptions("--define=foo=bar");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of("--define=foo=bar", ""),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    assertContainsPersistentMessage(
        executeFunction(key),
        EventKind.WARNING,
        /* frequency= */ 1,
        "also sets output-affecting flags in the command line or user bazelrc:"
            + " ['--define=foo=bar']");
  }

  @Test
  public void enforceCanonicalConfigs_compatiblePolicy_onlyDifferentValue_fails() throws Exception {
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
            name = "other_flag",
            build_setting_default = "default",
        )
        """);
    scratch.file(
        "test/PROJECT.scl",
        """
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          enforcement_policy = "compatible",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value", "--//test:other_flag=test_config_value"]
              ),
          ],
        )
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions =
        createBuildOptions("--//test:myflag=other_value", "--//test:other_flag=test_config_value");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(
                "--//test:myflag=other_value", "", "--//test:other_flag=test_config_value", ""),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown).hasMessageThat().contains("Found ['--//test:myflag=other_value']");
    assertThat(thrown).hasMessageThat().doesNotContain("['--//test:other_flag=test_config_value']");
  }

  @Test
  public void oldSchema_enforceCanonicalConfigs_wrongConfigsType() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/PROJECT.scl",
        """
        project = {
          "configs": 1,
        }
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions = createBuildOptions("--define=foo=bar");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains("configs variable must be a map of strings to lists of strings");
  }

  @Test
  public void oldSchema_enforceCanonicalConfigs_wrongConfigsKeyType() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/PROJECT.scl",
        """
        project = {
          "configs": {
            123: ["--compilation_mode=opt"],
          },
        }
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions = createBuildOptions("--define=foo=bar");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains("configs variable must be a map of strings to lists of strings");
  }

  @Test
  public void oldSchema_enforceCanonicalConfigs_wrongConfigsValueType() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/PROJECT.scl",
        """
        project = {
          "configs": {
            "test_config": 123,
          },
        }
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions = createBuildOptions("--define=foo=bar");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains("configs variable must be a map of strings to lists of strings");
  }

  @Test
  public void enforceCanonicalConfigsExtraFakeExpandedFlag_withSclConfig_fails() throws Exception {
    scratch.file(
        "test/build_settings.bzl",
        """
        string_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.string(flag = True),
            attrs = {
                "scope": attr.string(default = "universal"),
            },
        )
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
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          enforcement_policy = "strict",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value"]
              ),
              buildable_unit_pb2.BuildableUnit.create(
                  name = "other_config",
                  flags = ["--//test:myflag=other_config_value"]
              ),
          ],
        )
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            createBuildOptions(), // this is a fake flag so don't add it here.
            /* userOptions= */ ImmutableMap.of("--bar", "--config=foo"),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown).hasMessageThat().contains("Found ['--config=foo']");
  }

  @Test
  public void enforceCanonicalConfigs_extraFlagThatIsAlsoInConfig_differentValue_fails()
      throws Exception {
    scratch.file(
        "test/build_settings.bzl",
        """
        string_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.string(flag = True),
            attrs = {
                "scope": attr.string(default = "universal"),
            },
        )
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
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          enforcement_policy = "strict",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value"]
              ),
          ],
        )
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions = createBuildOptions("--//test:myflag=other_value");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of("--//test:myflag=other_value", ""),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown).hasMessageThat().contains("Found ['--//test:myflag=other_value']");
  }

  @Test
  public void enforceCanonicalConfigs_passedFlagThatIsInConfig_passes() throws Exception {
    scratch.file(
        "test/build_settings.bzl",
        """
        string_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.string(flag = True),
            attrs = {
                "scope": attr.string(default = "universal"),
            },
        )
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
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value"]
              ),
          ],
        )
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions = createBuildOptions("--//test:myflag=test_config_value");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of("--//test:myflag=test_config_value", ""),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var unused = executeFunction(key);
    assertDoesNotContainEvent("--scl_config must be the only configuration-affecting flag");
  }

  @Test
  public void enforceCanonicalConfigsExtraStarlarkFlag_fails() throws Exception {
    scratch.file(
        "test/build_settings.bzl",
        """
        string_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.string(flag = True),
            attrs = {
                "scope": attr.string(default = "universal"),
            },
        )
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
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          enforcement_policy = "strict",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value"]
              ),
              buildable_unit_pb2.BuildableUnit.create(
                  name = "other_config",
                  flags = ["--//test:myflag=other_config_value"]
              ),
          ],
        )
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions =
        createBuildOptions("--//test:starlark_flags_always_affect_configuration=yes_they_do");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(
                "--//test:starlark_flags_always_affect_configuration=yes_they_do", ""),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains("Found ['--//test:starlark_flags_always_affect_configuration=yes_they_do']");
  }

  @Test
  public void enforceCanonicalConfigsExtraTestFlag_passes() throws Exception {
    scratch.file(
        "test/build_settings.bzl",
        """
        string_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.string(flag = True),
            attrs = {
                "scope": attr.string(default = "universal"),
            },
        )
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
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          enforcement_policy = "strict",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value"]
              ),
              buildable_unit_pb2.BuildableUnit.create(
                  name = "other_config",
                  flags = ["--//test:myflag=other_config_value"]
              ),
          ],
        )
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions =
        createBuildOptions("--test_filter=foo", "--cache_test_results=true", "--test_arg=blah");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(
                "--test_filter=foo", "", "--cache_test_results=true", "", "--test_arg=blah", ""),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var unused = executeFunction(key);
    assertDoesNotContainEvent("--scl_config must be the only configuration-affecting flag");
  }

  @Test
  public void noEnforceCanonicalConfigs_noSclConfig_extraFlag_passes() throws Exception {
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
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value"]
              ),
              buildable_unit_pb2.BuildableUnit.create(
                  name = "other_config",
                  flags = ["--//test:myflag=other_config_value"]
              ),
          ],
        )
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions = createBuildOptions("--define=foo=bar");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "test_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of("--define=foo=bar", ""),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ false);

    var unused = executeFunction(key);
    assertDoesNotContainEvent("--scl_config must be the only configuration-affecting flag");
  }

  @Test
  public void enforceCanonicalConfigsNonExistentConfig_fails() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          enforcement_policy = "strict",
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value"]
              ),
          ],
        )
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            "non_existent_config",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
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
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--//test:myflag=test_config_value"]
              ),
              buildable_unit_pb2.BuildableUnit.create(
                  name = "other_config",
                  flags = ["--//test:myflag=other_config_value"]
              ),
          ],
        )
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains(
            "This project's builds must set --scl_config because no default config is defined");
  }

  @Test
  public void oldSchema_enforceCanonicalConfigsNonexistentDefaultConfig() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        project = {
          "configs": {
            "test_config": ['--//test:myflag=test_config_value'],
            "other_config": ['--//test:myflag=other_config_value'],
          },
            "default_config": "nonexistent_config",
        }
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains("default_config must be a string matching a configs variable definition");
  }

  @Test
  public void oldSchema_enforceCanonicalConfigs_wrongDefaultConfigType() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        project = {
          "configs": {
            "test_config": ['--//test:myflag=test_config_value'],
            "other_config": ['--//test:myflag=other_config_value'],
          },
          "default_config": ["test_config"],
        }
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains("default_config must be a string matching a configs variable definition");
  }

  @Test
  public void enforceCanonicalConfigsNoSclConfigFlagValidDefaultConfig() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
"""
load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
project = project_pb2.Project.create(
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--//test:myflag=test_config_value"],
          is_default = True,
      ),
  ],
)
""");
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);
    FlagSetValue flagSetsValue = executeFunction(key);

    assertThat(flagSetsValue.getOptionsFromFlagset())
        .containsExactly("--//test:myflag=test_config_value");
    assertContainsPersistentMessage(
        flagSetsValue,
        EventKind.INFO,
        /* frequency= */ 1,
        "Applying flags from the config 'test_config'");
  }

  @Test
  public void nonBuildOptions_areIgnored() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--bazelrc=foo"],
                  is_default = True,
              ),
          ],
        )
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);
    FlagSetValue flagSetsValue = executeFunction(key);

    assertThat(flagSetsValue.getOptionsFromFlagset()).isEmpty();
    assertDoesNotContainEvent("Applying flags from the config 'test_config'");
  }

  @Ignore("b/415352636")
  public void basicFlagsetFunctionalityWithTopLevelProjectSchema() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
        """
        load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
        project = project_pb2.Project.create(
          buildable_units = [
              buildable_unit_pb2.BuildableUnit.create(
                  name = "test_config",
                  flags = ["--bazelrc=foo"]
                  is_default = True,
              ),
          ],
        )
        """);
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ "",
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);
    FlagSetValue flagSetsValue = executeFunction(key);

    assertThat(flagSetsValue.getOptionsFromFlagset())
        .containsExactly("--//test:myflag=test_config_value");
    assertContainsPersistentMessage(
        flagSetsValue,
        EventKind.INFO,
        /* frequency= */ 1,
        "Applying flags from the config 'test_config'");
  }

  @Test
  public void clearUserDocumentationOfValidConfigs() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
"""
load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
project = project_pb2.Project.create(
  enforcement_policy = "strict",
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "debug",
          flags = ["--//test:myflag=debug_value"],
      ),
      buildable_unit_pb2.BuildableUnit.create(
          name = "release",
          flags = ["--//test:myflag=release_value"],
      ),
  ],
)
""");
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(Label.parseCanonical("//test:test_target")),
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ null,
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains(
            """
            This project's builds must set --scl_config because no default config is defined.

            This project supports:
              --scl_config=debug   -> [--//test:myflag=debug_value]
              --scl_config=release -> [--//test:myflag=release_value]

            This policy is defined in test/PROJECT.scl.
            """);
  }

  @Test
  public void buildingMultipleTargets_withSameConfig_isAllowed() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
"""
load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
project = project_pb2.Project.create(
  enforcement_policy = "warn",
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "default",
          flags = [],
          is_default = True,
      ),
      buildable_unit_pb2.BuildableUnit.create(
          name = "debug",
          flags = ["--//test:myflag=debug_value"],
      ),
      buildable_unit_pb2.BuildableUnit.create(
          name = "release",
          flags = ["--//test:myflag=release_value"],
      ),
  ],
)
""");
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(
                Label.parseCanonical("//test:test_target"),
                Label.parseCanonical("//test:test_target2")),
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ null,
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var unused = executeFunction(key);
    assertDoesNotContainEvent(
        "Cannot parse options: Building target(s) with different configurations are not supported");
  }

  @Test
  public void multipleTargetsWithMismatchingDefaultBuildableUnitsFails() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
"""
load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
project = project_pb2.Project.create(
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "foo_default",
          target_patterns = ["//test:test_target"],
          flags = ["--//test:myflag=foo_default_value"],
          is_default = True,
      ),
      buildable_unit_pb2.BuildableUnit.create(
          name = "bar_default",
          target_patterns = ["//test:test_target2"],
          flags = ["--//test:myflag=bar_default_value"],
          is_default = True,
      ),
  ],
)
""");
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(
                Label.parseCanonical("//test:test_target"),
                Label.parseCanonical("//test:test_target2")),
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ null,
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var thrown = assertThrows(Exception.class, () -> executeFunction(key));
    assertThat(thrown)
        .hasMessageThat()
        .contains("Building target(s) with different configurations are not supported");
  }

  @Test
  public void multipleTargetsWithDifferentDefaultsSucceedsIfSameFlags() throws Exception {
    createStringFlag("//test:myflag", /* defaultValue= */ "default");
    scratch.file(
        "test/PROJECT.scl",
"""
load("//test:project_proto.scl", "buildable_unit_pb2", "project_pb2")
project = project_pb2.Project.create(
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "foo_default",
          target_patterns = ["//test:test_target"],
          flags = ["--//test:myflag=common_default_value"],
          is_default = True,
      ),
      buildable_unit_pb2.BuildableUnit.create(
          name = "bar_default",
          target_patterns = ["//test:test_target2"],
          flags = ["--//test:myflag=common_default_value"],
          is_default = True,
      ),
  ],
)
""");
    BuildOptions buildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(
            ruleClassProvider.getFragmentRegistry().getOptionsClasses());
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");

    FlagSetValue.Key key =
        FlagSetValue.Key.create(
            ImmutableSet.of(
                Label.parseCanonical("//test:test_target"),
                Label.parseCanonical("//test:test_target2")),
            Label.parseCanonical("//test:PROJECT.scl"),
            /* sclConfig= */ null,
            buildOptions,
            /* userOptions= */ ImmutableMap.of(),
            /* configFlagDefinitions= */ ConfigFlagDefinitions.NONE,
            /* enforceCanonical= */ true);

    var unused = executeFunction(key);
    assertNoEvents();
  }

  private FlagSetValue executeFunction(FlagSetValue.Key key) throws Exception {
    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    EvaluationResult<FlagSetValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, /* keepGoing= */ false, reporter);
    if (result.hasError()) {
      throw result.getError(key).getException();
    }
    return result.get(key);
  }

  @Test
  @TestParameters(valuesProvider = TargetPatternProvider.class)
  public void doesBuildableUnitMatchTarget(
      boolean included, BuildableUnit buildableUnit, Label label) {
    assertThat(FlagSetFunction.doesBuildableUnitMatchTarget(buildableUnit, label))
        .isEqualTo(included);
  }

  static final class TargetPatternProvider extends TestParametersValuesProvider {

    private static TestParametersValues create(boolean included, String pattern, String label)
        throws Exception {
      return create(included, ImmutableList.of(pattern), label);
    }

    private static TestParametersValues create(
        boolean included, ImmutableList<String> patterns, String label) throws Exception {
      String name = String.format("%s-%s-%s", included ? "included" : "excluded", patterns, label);
      BuildableUnit buildableUnit =
          BuildableUnit.create(
              patterns, "Test Unit", ImmutableList.of("--flag"), /* isDefault= */ true);
      return TestParametersValues.builder()
          .name(name)
          .addParameter("included", included)
          .addParameter("buildableUnit", buildableUnit)
          .addParameter("label", Label.parseCanonicalUnchecked(label))
          .build();
    }

    @Override
    protected ImmutableList<TestParametersValues> provideValues(Context context) throws Exception {
      return ImmutableList.of(
          // Single pattern
          create(true, "//foo:foo", "//foo:foo"),
          create(false, "//foo:foo", "//foo:bar"),
          create(true, "//foo/...", "//foo:foo"),
          create(true, "//foo/...", "//foo/bar:bar"),
          create(false, "//foo/...", "//bar:bar"),
          create(false, "//foo/bar/...", "//foo:foo"),

          // Multiple patterns
          create(true, ImmutableList.of("//foo:foo", "//bar:bar"), "//foo:foo"),
          create(true, ImmutableList.of("//foo:foo", "//bar:bar"), "//bar:bar"),
          create(false, ImmutableList.of("//foo:foo", "//bar:bar"), "//quux:quux"),

          // Negative patterns
          create(false, "-//foo:foo", "//foo:foo"),
          create(false, "-//foo/...", "//foo:foo"),
          create(false, ImmutableList.of("//foo/...", "-//foo/bar/..."), "//foo/bar:bar"),
          create(true, ImmutableList.of("//foo/...", "-//foo/bar/..."), "//foo:foo"),
          create(
              true,
              ImmutableList.of("//foo/...", "-//foo/bar/...", "//foo/bar/baz/..."),
              "//foo/bar/baz"),
          create(
              true,
              ImmutableList.of("//foo/...", "-//foo/bar/...", "//foo/bar/baz/..."),
              "//foo:foo"),
          create(
              false,
              ImmutableList.of("//foo/...", "-//foo/bar/...", "//foo/bar/baz/..."),
              "//foo/bar/quux"));
    }
  }
}
