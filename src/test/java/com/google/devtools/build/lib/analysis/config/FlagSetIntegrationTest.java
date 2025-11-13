// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Integration tests for building with {@code --scl_config} (flag sets). See {@link Project} and
 * {@link CoreOptions#sclConfig} for details.
 *
 * <p>Use this for tests that cover building targets with {@code --scl-config}.
 *
 * <p>If you just want to test {@link com.google.devtools.build.lib.skyframe.config.FlagSetFunction}
 * (i.e. parsing --scl_config independent of how builds use it), use {@link
 * com.google.devtools.build.lib.skyframe.config.FlagSetFunctionTest}.
 *
 * <p>If you need full end-to-end testing, use {@code flagset_tests.sh}.
 */
@RunWith(TestParameterInjector.class)
public class FlagSetIntegrationTest extends BuildIntegrationTestCase {
  @Before
  public void setup() throws Exception {
    writeProjectSclDefinition("test/project_proto.scl", /* alsoWriteBuildFile= */ true);
  }

  /**
   * Given "//foo:myflag" and "default_value", creates the BUILD and .bzl files to realize a
   * string_flag with that label and default value.
   */
  private void createStringFlag(String labelName, String defaultValue) throws Exception {
    String flagDir = labelName.substring(2, labelName.indexOf(":"));
    String flagName = labelName.substring(labelName.indexOf(":") + 1);
    write(
        flagDir + "/build_settings.bzl",
"""
string_flag = rule(implementation = lambda ctx: [], build_setting = config.string(flag = True))
""");
    write(
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

  /** Given ""//test:s", creates the BUILD and .bzl files for a trivial target with that label. */
  private void createSimpleTarget(String labelName) throws Exception {
    String targetDir = labelName.substring(2, labelName.indexOf(":"));
    String targetName = labelName.substring(labelName.indexOf(":") + 1);
    write(
        targetDir + "/defs.bzl",
"""
simple_rule = rule(
 implementation = lambda ctx: [],
 attrs = {}
 )
""");
    write(
        targetDir + "/BUILD",
"""
load(":defs.bzl", "simple_rule")
simple_rule(name = "%s")
"""
            .formatted(targetName));
  }

  @Test
  public void noSclConfigSetAndNoDefaultConfig(@TestParameter boolean enforceProjectConfigs)
      throws Exception {
    createSimpleTarget("//test:s");
    write(
        "test/PROJECT.scl",
"""
load(
  "//test:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
    enforcement_policy = "warn",
    buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--//test:myflag=test_config_value"],
          is_default = False,
      )
  ],
)
""");

    addOptions("--enforce_project_configs=" + (enforceProjectConfigs ? "1" : "0"));
    if (!enforceProjectConfigs) {
      // There's no default project config but that doesn't matter when project enforcement is
      // disabled: the entire PROJECT.scl is ignored.
      assertThat(buildTarget("//test:s")).isNotNull();
    } else {
      // With project enforcement enabled, no default config means user must set --scl_config.
      InvalidConfigurationException expectedError =
          assertThrows(InvalidConfigurationException.class, () -> buildTarget("//test:s"));
      assertThat(expectedError)
          .hasMessageThat()
          .contains("This project's builds must set --scl_config");
    }
  }

  @Test
  public void warnModeAddsBothUserAndProjectStarlarkFlags() throws Exception {
    createStringFlag("//test1:project_flag", /* defaultValue= */ "default");
    createStringFlag("//test2:user_flag", /* defaultValue= */ "default");
    createSimpleTarget("//test:s");
    write(
        "test/PROJECT.scl",
"""
load(
  "//test:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
    enforcement_policy = "warn",
    buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--//test1:project_flag=set_by_project"],
          is_default = True,
      )
  ],
)
""");

    addOptions("--enforce_project_configs=1", "--//test2:user_flag=set_by_user");
    var result = buildTarget("//test:s");

    assertThat(result).isNotNull();
    assertThat(result.getBuildConfiguration().getOptions().getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//test1:project_flag"),
            "set_by_project",
            Label.parseCanonicalUnchecked("//test2:user_flag"),
            "set_by_user");
  }

  @Test
  public void warnMode_userFlagTakesPrecedenceOverProjectFlag() throws Exception {
    createStringFlag("//test1:flag", /* defaultValue= */ "default");
    createSimpleTarget("//test:s");
    write(
        "test/PROJECT.scl",
"""
load(
  "//test:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
    enforcement_policy = "warn",
    buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--//test1:flag=set_by_project"],
          is_default = True,
      )
  ],
)
""");

    addOptions("--enforce_project_configs=1", "--//test1:flag=set_by_user");
    var result = buildTarget("//test:s");

    assertThat(result).isNotNull();
    assertThat(result.getBuildConfiguration().getOptions().getStarlarkOptions())
        .containsExactly(Label.parseCanonicalUnchecked("//test1:flag"), "set_by_user");
  }

  @Test
  public void warnMode_allowMultipleFlagShowsUserSettingsLast() throws Exception {
    write(
        "test1//build_settings.bzl",
"""
repeatable_string_flag = rule(
    implementation = lambda ctx: [],
    build_setting = config.string_list(flag = True, repeatable = True),
)
""");
    write(
        "test1/BUILD",
        """
        load(":build_settings.bzl", "repeatable_string_flag")
        repeatable_string_flag(
            name = "flag",
            build_setting_default = [],
        )
        """);
    createSimpleTarget("//test:s");
    write(
        "test/PROJECT.scl",
"""
load(
  "//test:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
    enforcement_policy = "warn",
    buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--//test1:flag=set_by_project"],
          is_default = True,
      )
  ],
)
""");

    addOptions("--enforce_project_configs=1", "--//test1:flag=set_by_user");
    var result = buildTarget("//test:s");

    assertThat(result).isNotNull();
    assertThat(result.getBuildConfiguration().getOptions().getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//test1:flag"),
            ImmutableList.of("set_by_project", "set_by_user"));
  }
}
