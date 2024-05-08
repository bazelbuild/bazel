// Copyright 2017 The Bazel Authors. All rights reserved.
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
// limitations under the License

package com.google.devtools.build.lib.rules.config;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the config_feature_flag rule. */
@RunWith(JUnit4.class)
public final class ConfigFeatureFlagNamingTest extends BuildViewTestCase {

  private String getMnemonic(ConfiguredTarget target) {
    return getConfiguration(target).getMnemonic();
  }

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder().addRuleDefinition(new FeatureFlagSetterRule());
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Test
  public void featureFlagSetter_sameSettingYieldsSameMnemonic_legacy() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        feature_flag_setter(
            name = "top_a",
            exports_flag = ":flag",
            flag_values = {
                ":flag": "configured",
            },
            transitive_configs = [":flag"],
        )

        feature_flag_setter(
            name = "top_b",
            exports_flag = ":flag",
            flag_values = {
                ":flag": "configured",
            },
            transitive_configs = [":flag"],
        )

        config_feature_flag(
            name = "flag",
            allowed_values = [
                "default",
                "configured",
                "other",
            ],
            default_value = "default",
        )
        """);
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    String aMnemonic = getMnemonic(getConfiguredTarget("//test:top_a"));
    String bMnemonic = getMnemonic(getConfiguredTarget("//test:top_b"));
    assertThat(aMnemonic).isEqualTo(bMnemonic);
  }

  @Test
  public void featureFlagSetter_diffSettingYieldsDiffMnemonic_legacy() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        feature_flag_setter(
            name = "top_a",
            exports_flag = ":flag",
            flag_values = {
                ":flag": "configured",
            },
            transitive_configs = [":flag"],
        )

        feature_flag_setter(
            name = "top_b",
            exports_flag = ":flag",
            flag_values = {
                ":flag": "other",
            },
            transitive_configs = [":flag"],
        )

        config_feature_flag(
            name = "flag",
            allowed_values = [
                "default",
                "configured",
                "other",
            ],
            default_value = "default",
        )
        """);
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    String aMnemonic = getMnemonic(getConfiguredTarget("//test:top_a"));
    String bMnemonic = getMnemonic(getConfiguredTarget("//test:top_b"));
    assertThat(aMnemonic).isNotEqualTo(bMnemonic);
  }

  @Test
  public void featureFlagSetter_sameSettingYieldsSameMnemonic_diff() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        feature_flag_setter(
            name = "top_a",
            exports_flag = ":flag",
            flag_values = {
                ":flag": "configured",
            },
            transitive_configs = [":flag"],
        )

        feature_flag_setter(
            name = "top_b",
            exports_flag = ":flag",
            flag_values = {
                ":flag": "configured",
            },
            transitive_configs = [":flag"],
        )

        config_feature_flag(
            name = "flag",
            allowed_values = [
                "default",
                "configured",
                "other",
            ],
            default_value = "default",
        )
        """);
    useConfiguration(
        "--enforce_transitive_configs_for_config_feature_flag",
        "--experimental_output_directory_naming_scheme=diff_against_dynamic_baseline");
    String aMnemonic = getMnemonic(getConfiguredTarget("//test:top_a"));
    String bMnemonic = getMnemonic(getConfiguredTarget("//test:top_b"));
    assertThat(aMnemonic).isEqualTo(bMnemonic);
  }

  @Test
  public void featureFlagSetter_diffSettingYieldsDiffMnemonic_diff() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        feature_flag_setter(
            name = "top_a",
            exports_flag = ":flag",
            flag_values = {
                ":flag": "configured",
            },
            transitive_configs = [":flag"],
        )

        feature_flag_setter(
            name = "top_b",
            exports_flag = ":flag",
            flag_values = {
                ":flag": "other",
            },
            transitive_configs = [":flag"],
        )

        config_feature_flag(
            name = "flag",
            allowed_values = [
                "default",
                "configured",
                "other",
            ],
            default_value = "default",
        )
        """);
    useConfiguration(
        "--enforce_transitive_configs_for_config_feature_flag",
        "--experimental_output_directory_naming_scheme=diff_against_dynamic_baseline");
    String aMnemonic = getMnemonic(getConfiguredTarget("//test:top_a"));
    String bMnemonic = getMnemonic(getConfiguredTarget("//test:top_b"));
    assertThat(aMnemonic).isNotEqualTo(bMnemonic);
  }

  @Test
  public void untrimmedFlag_doesNothing_legacy() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        feature_flag_setter(
            name = "via_setter",
            exports_flag = ":flag",
            flag_values = {
                ":flag": "configured",
            },
            transitive_configs = [":flag"],
            deps = [":via_consumer"],
        )

        genrule(
            name = "via_consumer",
            outs = ["out"],
            cmd = "touch $@",
            transitive_configs = [":flag"],
        )

        config_feature_flag(
            name = "flag",
            allowed_values = [
                "default",
                "configured",
                "other",
            ],
            default_value = "default",
        )
        """);
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    ConfiguredTarget viaSetter = getConfiguredTarget("//test:via_setter");
    ConfiguredTarget viaConsumer = getDirectPrerequisite(viaSetter, "//test:via_consumer");
    assertThat(getMnemonic(viaSetter)).isEqualTo(getMnemonic(viaConsumer));
  }

  @Test
  public void trimmedFlag_causesDiff_legacy() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        feature_flag_setter(
            name = "via_setter",
            exports_flag = ":flag",
            flag_values = {
                ":flag": "configured",
            },
            transitive_configs = [":flag"],
            deps = [":via_consumer"],
        )

        genrule(
            name = "via_consumer",
            outs = ["out"],
            cmd = "touch $@",
            transitive_configs = [],
        )

        config_feature_flag(
            name = "flag",
            allowed_values = [
                "default",
                "configured",
                "other",
            ],
            default_value = "default",
        )
        """);
    useConfiguration("--enforce_transitive_configs_for_config_feature_flag");
    ConfiguredTarget viaSetter = getConfiguredTarget("//test:via_setter");
    ConfiguredTarget viaConsumer = getDirectPrerequisite(viaSetter, "//test:via_consumer");
    assertThat(getMnemonic(viaSetter)).isNotEqualTo(getMnemonic(viaConsumer));
  }

  @Test
  public void untrimmedFlag_doesNothing_diff() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        feature_flag_setter(
            name = "via_setter",
            exports_flag = ":flag",
            flag_values = {
                ":flag": "configured",
            },
            transitive_configs = [":flag"],
            deps = [":via_consumer"],
        )

        genrule(
            name = "via_consumer",
            outs = ["out"],
            cmd = "touch $@",
            transitive_configs = [":flag"],
        )

        config_feature_flag(
            name = "flag",
            allowed_values = [
                "default",
                "configured",
                "other",
            ],
            default_value = "default",
        )
        """);
    useConfiguration(
        "--enforce_transitive_configs_for_config_feature_flag",
        "--experimental_output_directory_naming_scheme=diff_against_dynamic_baseline");
    ConfiguredTarget viaSetter = getConfiguredTarget("//test:via_setter");
    ConfiguredTarget viaConsumer = getDirectPrerequisite(viaSetter, "//test:via_consumer");
    assertThat(getMnemonic(viaSetter)).isEqualTo(getMnemonic(viaConsumer));
  }

  @Test
  public void trimmedFlag_causesDiff_diff() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        feature_flag_setter(
            name = "via_setter",
            exports_flag = ":flag",
            flag_values = {
                ":flag": "configured",
            },
            transitive_configs = [":flag"],
            deps = [":via_consumer"],
        )

        genrule(
            name = "via_consumer",
            outs = ["out"],
            cmd = "touch $@",
            transitive_configs = [],
        )

        config_feature_flag(
            name = "flag",
            allowed_values = [
                "default",
                "configured",
                "other",
            ],
            default_value = "default",
        )
        """);
    useConfiguration(
        "--enforce_transitive_configs_for_config_feature_flag",
        "--experimental_output_directory_naming_scheme=diff_against_dynamic_baseline");
    ConfiguredTarget viaSetter = getConfiguredTarget("//test:via_setter");
    ConfiguredTarget viaConsumer = getDirectPrerequisite(viaSetter, "//test:via_consumer");
    assertThat(getMnemonic(viaSetter)).isNotEqualTo(getMnemonic(viaConsumer));
  }
}
