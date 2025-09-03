// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.starlark;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test of common logic between Starlark-defined transitions. Rule-transition- or
 * attr-transition-specific logic should be tested in {@link StarlarkRuleTransitionProviderTest} and
 * {@link StarlarkAttrTransitionProviderTest}.
 */
@RunWith(JUnit4.class)
public class StarlarkTransitionTest extends BuildViewTestCase {

  /** Extra options for this test. */
  public static class DummyTestOptions extends FragmentOptions {
    public DummyTestOptions() {}

    @Option(
        name = "non_configurable_option",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "non-configurable",
        metadataTags = {OptionMetadataTag.NON_CONFIGURABLE})
    public String nonConfigurableOption;

    @Option(
        name = "disallowed_option",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "default")
    public String disallowedOption;
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

  static void writeAllowlistFile(Scratch scratch) throws Exception {
    scratch.overwriteFile(
        "tools/allowlists/function_transition_allowlist/BUILD",
        """
        package_group(
            name = "function_transition_allowlist",
            packages = [
                "//test/...",
            ],
        )
        """);
  }

  @Test
  public void testDupeSettingsInInputsThrowsError() throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _setting_impl(ctx):
            return []

        string_flag = rule(
            implementation = _setting_impl,
            build_setting = config.string(flag = True),
        )

        def _transition_impl(settings, attr):
            return {"//test:formation": "mesa"}

        formation_transition = transition(
            implementation = _transition_impl,
            inputs = ["@//test:formation", "//test:formation"],  # duplicates here
            outputs = ["//test:formation"],
        )

        def _impl(ctx):
            return []

        state = rule(
            implementation = _impl,
            cfg = formation_transition,
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "state", "string_flag")

        state(name = "arizona")

        string_flag(
            name = "formation",
            build_setting_default = "canyon",
        )
        """);

    reporter.removeHandler(failFastHandler);

    getConfiguredTarget("//test:arizona");
    assertContainsEvent(
        "Transition declares duplicate build setting '@@//test:formation' in INPUTS");
  }

  @Test
  public void testDupeSettingsInOutputsThrowsError() throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _setting_impl(ctx):
            return []

        string_flag = rule(
            implementation = _setting_impl,
            build_setting = config.string(flag = True),
        )

        def _transition_impl(settings, attr):
            return {"//test:formation": "mesa"}

        formation_transition = transition(
            implementation = _transition_impl,
            inputs = ["//test:formation"],
            outputs = ["@//test:formation", "//test:formation"],  # duplicates here
        )

        def _impl(ctx):
            return []

        state = rule(
            implementation = _impl,
            cfg = formation_transition,
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "state", "string_flag")

        state(name = "arizona")

        string_flag(
            name = "formation",
            build_setting_default = "canyon",
        )
        """);

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:arizona");
    assertContainsEvent(
        "Transition declares duplicate build setting '@@//test:formation' in OUTPUTS");
  }

  @Test
  public void testDifferentFormsOfFlagInInputsAndOutputs() throws Exception {
    writeAllowlistFile(scratch);
    scratch.file(
        "test/defs.bzl",
        """
        def _setting_impl(ctx):
            return []

        string_flag = rule(
            implementation = _setting_impl,
            build_setting = config.string(flag = True),
        )

        def _transition_impl(settings, attr):
            formation = settings["@//test:formation"]
            if formation.endswith("-transitioned"):
                new_value = formation
            else:
                new_value = formation + "-transitioned"
            return {
                "//test:formation": new_value,
            }

        formation_transition = transition(
            implementation = _transition_impl,
            inputs = ["@//test:formation"],
            outputs = ["//test:formation"],
        )

        def _impl(ctx):
            return []

        state = rule(
            implementation = _impl,
            cfg = formation_transition,
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "state", "string_flag")

        state(name = "arizona")

        string_flag(
            name = "formation",
            build_setting_default = "canyon",
        )
        """);

    Map<Label, Object> starlarkOptions =
        getConfiguration(getConfiguredTarget("//test:arizona")).getOptions().getStarlarkOptions();
    assertThat(starlarkOptions).hasSize(1);
    assertThat(starlarkOptions.get(Label.parseCanonicalUnchecked("//test:formation")))
        .isEqualTo("canyon-transitioned");
  }

  private void writeDefBzlWithStringFlagAndEaterRule() throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _setting_impl(ctx):
            return []

        string_flag = rule(
            implementation = _setting_impl,
            build_setting = config.string(flag = True),
        )

        def _transition_impl(settings, attr):
            if settings["@//options:fruit"].endswith("-eaten"):
                return {"//options:fruit": settings["@//options:fruit"]}
            return {"//options:fruit": settings["@//options:fruit"] + "-eaten"}

        eating_transition = transition(
            implementation = _transition_impl,
            inputs = ["@//options:fruit"],
            outputs = ["//options:fruit"],
        )

        def _impl(ctx):
            return []

        eater = rule(
            implementation = _impl,
            cfg = eating_transition,
        )
        """);
  }

  @Test
  public void testDifferentDefaultsRerunsTransitionTest() throws Exception {
    writeAllowlistFile(scratch);
    writeDefBzlWithStringFlagAndEaterRule();
    scratch.file(
        "options/BUILD",
        """
        load("//test:defs.bzl", "string_flag")

        string_flag(
            name = "fruit",
            build_setting_default = "apple",
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "eater")

        eater(name = "foo")
        """);

    assertThat(
            getConfiguration(getConfiguredTarget("//test:foo"))
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseCanonicalUnchecked("//options:fruit")))
        .isEqualTo("apple-eaten");

    scratch.overwriteFile(
        "options/BUILD",
        """
        load("//test:defs.bzl", "string_flag")

        string_flag(
            name = "fruit",
            build_setting_default = "orange",
        )
        """);
    invalidatePackages();
    assertThat(
            getConfiguration(getConfiguredTarget("//test:foo"))
                .getOptions()
                .getStarlarkOptions()
                .get(Label.parseCanonicalUnchecked("//options:fruit")))
        .isEqualTo("orange-eaten");
  }

  @Test
  public void testAliasChangeRerunsTransitionTest() throws Exception {
    writeAllowlistFile(scratch);
    writeDefBzlWithStringFlagAndEaterRule();
    scratch.file(
        "options/BUILD",
        """
        load("//test:defs.bzl", "string_flag")

        string_flag(
            name = "usually_apple",
            build_setting_default = "apple",
        )

        string_flag(
            name = "usually_orange",
            build_setting_default = "orange",
        )

        alias(
            name = "fruit",
            actual = ":usually_apple",
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "eater")

        eater(name = "foo")
        """);

    assertThat(
            getConfiguration(getConfiguredTarget("//test:foo")).getOptions().getStarlarkOptions())
        .containsExactly(Label.parseCanonicalUnchecked("//options:usually_apple"), "apple-eaten");

    scratch.overwriteFile(
        "options/BUILD",
        """
        load("//test:defs.bzl", "string_flag")

        string_flag(
            name = "usually_apple",
            build_setting_default = "apple",
        )

        string_flag(
            name = "usually_orange",
            build_setting_default = "orange",
        )

        alias(
            name = "fruit",
            actual = ":usually_orange",
        )
        """);
    invalidatePackages();

    assertThat(
            getConfiguration(getConfiguredTarget("//test:foo")).getOptions().getStarlarkOptions())
        .containsExactly(Label.parseCanonicalUnchecked("//options:usually_orange"), "orange-eaten");
  }

  @Test
  public void testChangingNonConfigurableOptionFails() throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _transition_impl(settings, attr):
            return {"//command_line_option:non_configurable_option": "something_else"}

        _transition = transition(
            implementation = _transition_impl,
            inputs = [],
            outputs = ["//command_line_option:non_configurable_option"],
        )

        def _impl(ctx):
            return []

        state = rule(
            implementation = _impl,
            cfg = _transition,
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "state")

        state(name = "arizona")
        """);

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:arizona");
    assertContainsEvent(
        "transition outputs [//command_line_option:non_configurable_option] cannot be changed: they"
            + " are non-configurable");
  }

  @Test
  public void testNonConfigurableOptionAsTransitionInputFails() throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _transition_impl(settings, attr):
            return {}

        _transition = transition(
            implementation = _transition_impl,
            inputs = ["//command_line_option:non_configurable_option"],
            outputs = [],
        )

        def _impl(ctx):
            return []

        state = rule(
            implementation = _impl,
            cfg = _transition,
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "state")

        state(name = "arizona")
        """);

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:arizona");
    assertContainsEvent(
        "transition inputs [//command_line_option:non_configurable_option] cannot be changed: they"
            + " are non-configurable");
  }

  @Test
  public void testDisallowedOptionInTransitionInputsFails() throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _transition_impl(settings, attr):
            return {}

        _transition = transition(
            implementation = _transition_impl,
            inputs = ["//command_line_option:disallowed_option"],
            outputs = [],
        )

        def _impl(ctx):
            return []
        simple_rule = rule(
            implementation = _impl,
            cfg = _transition,
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "simple_rule")
        simple_rule(name = "t1")
        """);
    setBuildLanguageOptions("--incompatible_disable_transitions_on=disallowed_option");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:t1");
    assertContainsEvent("Option 'disallowed_option' is not allowed in transitions INPUTS");
  }

  @Test
  public void testDisallowedOptionInTransitionOutputsFails() throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _transition_impl(settings, attr):
            return {}

        _transition = transition(
            implementation = _transition_impl,
            inputs = [],
            outputs = ["//command_line_option:disallowed_option"],
        )

        def _impl(ctx):
            return []
        simple_rule = rule(
            implementation = _impl,
            cfg = _transition,
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "simple_rule")
        simple_rule(name = "t1")
        """);
    setBuildLanguageOptions("--incompatible_disable_transitions_on=disallowed_option");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:t1");
    assertContainsEvent("Option 'disallowed_option' is not allowed in transitions OUTPUTS");
  }
}
