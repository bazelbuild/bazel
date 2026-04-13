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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.skyframe.NodeEntry;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Test of common logic between Starlark-defined transitions. Rule-transition- or
 * attr-transition-specific logic should be tested in {@link StarlarkRuleTransitionProviderTest} and
 * {@link StarlarkAttrTransitionProviderTest}.
 */
@RunWith(TestParameterInjector.class)
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

    @Option(
        name = "existing_flag",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "native_default_value")
    public String existingFlag;
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

  private static void writeAllowlistFile(Scratch scratch) throws Exception {
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
    assertContainsEvent("Transition declares duplicate build setting '//test:formation' in INPUTS");
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
        "Transition declares duplicate build setting '//test:formation' in OUTPUTS");
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

  private void writeTestDefForFlagAlias(String nativeFlagName, boolean defaultValue)
      throws Exception {
    writeAllowlistFile(scratch);
    String value =
        defaultValue
            ? "'foo_starlark_default_value'"
            : String.format(
                "'transitioned ' + settings['//command_line_option:%s']", nativeFlagName);

    scratch.file(
        "test/defs.bzl",
        String.format(
            """
            def _setting_impl(ctx):
                return []

            my_flag = rule(
                implementation = _setting_impl,
                build_setting = config.string(flag = True),
            )

            def _transition_impl(settings, attr):
                return {
                    "//command_line_option:%s": %s,
                }

            my_transition = transition(
                implementation = _transition_impl,
                inputs = ["//command_line_option:%s"],
                outputs = ["//command_line_option:%s"],
            )

            def _impl(ctx):
                return []

            my_rule = rule(
                implementation = _impl,
                cfg = my_transition,
            )
            """,
            nativeFlagName, value, nativeFlagName, nativeFlagName));
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "my_rule", "my_flag")

        my_rule(name = "t1")

        my_flag(
            name = "foo_starlark",
            build_setting_default = "foo_starlark_default_value",
        )

        my_flag(
            name = "bar_starlark",
            build_setting_default = "bar_starlark_default_value",
        )
        """);
  }

  @Test
  public void testStarlarkFlagWithAliasInTransition(
      @TestParameter({"existing_flag", "new_alias"}) String nativeFlagName) throws Exception {
    writeTestDefForFlagAlias(nativeFlagName, /* defaultValue= */ false);

    useConfiguration(
        String.format("--flag_alias=%s=//test:foo_starlark", nativeFlagName),
        String.format("--%s=cmd_flag_value", nativeFlagName));
    var fooOptions = getConfiguration(getConfiguredTarget("//test:t1")).getOptions();
    assertThat(fooOptions.getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//test:foo_starlark"), "transitioned cmd_flag_value");
    if (nativeFlagName.equals("existing_flag")) {
      // The native flag value should not change.
      assertThat(fooOptions.get(DummyTestOptions.class).existingFlag)
          .isEqualTo("native_default_value");
    }

    // Modify the flag alias to point to //test:bar_starlark and make sure the transition updates
    // the new flag value.
    useConfiguration(
        String.format("--flag_alias=%s=//test:bar_starlark", nativeFlagName),
        String.format("--%s=cmd_flag_value", nativeFlagName));
    var barOptions = getConfiguration(getConfiguredTarget("//test:t1")).getOptions();
    assertThat(barOptions.getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//test:bar_starlark"), "transitioned cmd_flag_value");
    if (nativeFlagName.equals("existing_flag")) {
      // The native flag value should not change.
      assertThat(barOptions.get(DummyTestOptions.class).existingFlag)
          .isEqualTo("native_default_value");
    }
  }

  @Test
  public void testDefaultStarlarkFlagValue_passedToAlias(
      @TestParameter({"existing_flag", "new_alias"}) String nativeFlagName) throws Exception {
    writeTestDefForFlagAlias(nativeFlagName, /* defaultValue= */ false);

    useConfiguration(String.format("--flag_alias=%s=//test:foo_starlark", nativeFlagName));
    var fooOptions = getConfiguration(getConfiguredTarget("//test:t1")).getOptions();
    assertThat(fooOptions.getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//test:foo_starlark"),
            "transitioned foo_starlark_default_value");

    // Modify the flag alias to point to //test:bar_starlark and make sure the transition sees and
    // transitions the new flag value.
    useConfiguration(String.format("--flag_alias=%s=//test:bar_starlark", nativeFlagName));
    var barOptions = getConfiguration(getConfiguredTarget("//test:t1")).getOptions();
    assertThat(barOptions.getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//test:bar_starlark"),
            "transitioned bar_starlark_default_value");
  }

  @Test
  public void testWritingDefaultValueToStarlarkFlag_removedFromBuildOptions(
      @TestParameter({"existing_flag", "new_alias"}) String nativeFlagName) throws Exception {
    writeTestDefForFlagAlias(nativeFlagName, /* defaultValue= */ true);
    useConfiguration(String.format("--flag_alias=%s=//test:foo_starlark", nativeFlagName));

    var options = getConfiguration(getConfiguredTarget("//test:t1")).getOptions();

    assertThat(options.getStarlarkOptions()).isEmpty();
  }

  @Test
  public void testStarlarkFlagAndAliasInInputs_haveSameValue(
      @TestParameter({"existing_flag", "new_alias"}) String nativeFlagName) throws Exception {
    writeAllowlistFile(scratch);
    scratch.file(
        "test/defs.bzl",
        String.format(
            """
            def _setting_impl(ctx):
                return []

            my_flag = rule(
                implementation = _setting_impl,
                build_setting = config.string(flag = True),
            )

            def _transition_impl(settings, attr):
                if settings["//test:foo_starlark"] != settings["//command_line_option:%s"]:
                    fail("Starlark flag '@@//test:foo_starlark' and its alias '//command_line_option:%s' have different values: '{}' and '{}'".format(
                        settings["//test:foo_starlark"],
                        settings["//command_line_option:%s"],
                    ))
                return {}

            my_transition = transition(
                implementation = _transition_impl,
                inputs = ["//test:foo_starlark", "//command_line_option:%s"],
                outputs = [],
            )

            def _impl(ctx):
                return []

            my_rule = rule(
                implementation = _impl,
                cfg = my_transition,
            )
            """,
            nativeFlagName, nativeFlagName, nativeFlagName, nativeFlagName));
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "my_rule", "my_flag")

        my_rule(name = "t1")

        my_flag(
            name = "foo_starlark",
            build_setting_default = "starlark_default_value",
        )
        """);
    useConfiguration(
        String.format("--flag_alias=%s=//test:foo_starlark", nativeFlagName),
        String.format("--%s=cmd_flag_value", nativeFlagName));

    var unused = getConfiguredTarget("//test:t1");

    assertNoEvents();
  }

  @Test
  public void testTransitionWritesDifferentValueToFlagAndAlias_notAllowed(
      @TestParameter({"existing_flag", "new_alias"}) String nativeFlagName) throws Exception {
    writeAllowlistFile(scratch);
    scratch.file(
        "test/defs.bzl",
        String.format(
            """
            def _setting_impl(ctx):
                return []

            my_flag = rule(
                implementation = _setting_impl,
                build_setting = config.string(flag = True),
            )

            def _transition_impl(settings, attr):
                return {
                    "//command_line_option:%s": "val_for_native",
                    "//test:foo_starlark": "val_for_starlark",
                }

            my_transition = transition(
                implementation = _transition_impl,
                inputs = [],
                outputs = ["//test:foo_starlark", "//command_line_option:%s"],
            )

            def _impl(ctx):
                return []

            my_rule = rule(
                implementation = _impl,
                cfg = my_transition,
            )
            """,
            nativeFlagName, nativeFlagName));
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "my_rule", "my_flag")

        my_rule(name = "t1")

        my_flag(
            name = "foo_starlark",
            build_setting_default = "starlark_default_value",
        )
        """);
    useConfiguration(
        String.format("--flag_alias=%s=//test:foo_starlark", nativeFlagName),
        String.format("--%s=cmd_flag_value", nativeFlagName));

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:t1");
    assertContainsEvent(
        String.format(
            "Starlark flag '//test:foo_starlark' and its alias '//command_line_option:%s'"
                + " have different values: 'val_for_starlark' and 'val_for_native'",
            nativeFlagName));
  }

  private void writeExecTransition(String nativeFlagName) throws Exception {
    writeAllowlistFile(scratch);
    scratch.file(
        "test/defs.bzl",
        String.format(
            """
            def _setting_impl(ctx):
                return []

            my_flag = rule(
                implementation = _setting_impl,
                build_setting = config.string(flag = True),
            )

            def _transition_impl(settings, attr):
                return {
                    "//command_line_option:%s": 'transitioned ' + settings['//command_line_option:%s'],
                    "//command_line_option:is exec configuration": True,
                }

            my_transition = transition(
                implementation = _transition_impl,
                inputs = ["//command_line_option:%s"],
                outputs = [
                  "//command_line_option:%s",
                  "//command_line_option:is exec configuration",
                ],
            )

            def _impl(ctx):
                return []

            my_rule = rule(
                implementation = _impl,
                attrs = {'dep': attr.label(cfg = 'exec')},
            )
            """,
            nativeFlagName, nativeFlagName, nativeFlagName, nativeFlagName));
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "my_rule", "my_flag")

        my_rule(name = "t1", dep = ':t2')
        my_rule(name = "t2")

        my_flag(
          name = "foo_starlark",
          build_setting_default = "foo_starlark_default_value",
        )
        """);
  }

  @Test
  public void testStarlarkFlagAliasNotUsedInExecTransition_existingNativeFlag_pass(
      @TestParameter boolean isAlias, @TestParameter boolean starlarkFlagHasValue)
      throws Exception {
    writeExecTransition("existing_flag");
    List<String> args = new ArrayList<>();
    args.add("--experimental_exec_config=//test:defs.bzl%my_transition");
    if (isAlias) {
      args.add("--flag_alias=existing_flag=//test:foo_starlark");
    }
    if (starlarkFlagHasValue) {
      args.add("--//test:foo_starlark=cmd_value");
    }
    useConfiguration(args.toArray(new String[0]));

    getConfiguredTarget("//test:t1");

    var baselineExecConfig = execConfig;
    assertThat(baselineExecConfig.getOptions().get(DummyTestOptions.class).existingFlag)
        .isEqualTo("transitioned native_default_value");
    if (starlarkFlagHasValue) {
      assertThat(baselineExecConfig.getOptions().getStarlarkOptions())
          .containsExactly(Label.parseCanonicalUnchecked("//test:foo_starlark"), "cmd_value");
    } else {
      assertThat(baselineExecConfig.getOptions().getStarlarkOptions()).isEmpty();
    }

    var t2ExecConfig =
        getConfiguration(Iterables.getOnlyElement(getComputedConfiguredTarget("//test:t2")));
    assertThat(t2ExecConfig.getOptions().get(DummyTestOptions.class).existingFlag)
        .isEqualTo("transitioned native_default_value");
    if (starlarkFlagHasValue) {
      assertThat(t2ExecConfig.getOptions().getStarlarkOptions())
          .containsExactly(Label.parseCanonicalUnchecked("//test:foo_starlark"), "cmd_value");
    } else {
      assertThat(t2ExecConfig.getOptions().getStarlarkOptions()).isEmpty();
    }
  }

  @Test
  public void testStarlarkFlagAliasNotUsedInExecTransition_nonExistingNativeFlag_fail(
      @TestParameter boolean isAlias, @TestParameter boolean starlarkFlagHasValue)
      throws Exception {
    writeExecTransition("new_flag");
    List<String> args = new ArrayList<>();
    args.add("--experimental_exec_config=//test:defs.bzl%my_transition");
    if (isAlias) {
      args.add("--flag_alias=new_flag=//test:foo_starlark");
    }
    if (starlarkFlagHasValue) {
      args.add("--//test:foo_starlark=cmd_value");
    }

    AssertionError e =
        assertThrows(AssertionError.class, () -> useConfiguration(args.toArray(new String[0])));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "transition inputs [//command_line_option:new_flag] do not correspond to valid"
                + " settings");
  }

  @Test
  public void testTransitionUsesAliasesInExecAndNonExecTransitions() throws Exception {
    writeAllowlistFile(scratch);
    scratch.file(
        "test/defs.bzl",
        """
        def _setting_impl(ctx):
            return []

        my_flag = rule(
            implementation = _setting_impl,
            build_setting = config.string(flag = True),
        )

        def _transition_impl(settings, attr):
            return {
                "//command_line_option:existing_flag": "transitioned " + settings["//command_line_option:existing_flag"],
                "//test:foo_starlark": "transitioned " + settings["//test:foo_starlark"],
                "//command_line_option:is exec configuration": True,
            }

        my_transition = transition(
            implementation = _transition_impl,
            inputs = ["//test:foo_starlark", "//command_line_option:existing_flag"],
            outputs = ["//test:foo_starlark",
                 "//command_line_option:existing_flag",
                 "//command_line_option:is exec configuration"],
        )

        def _impl(ctx):
            return []

        my_rule = rule(
            implementation = _impl,
            attrs = {
                'exec_dep': attr.label(cfg = 'exec'),
                'non_exec_dep': attr.label(cfg = my_transition),
            },
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "my_rule", "my_flag")

        my_rule(name = "t1", exec_dep = ":t2", non_exec_dep = ":t3")
        my_rule(name = "t2")
        my_rule(name = "t3")

        my_flag(
            name = "foo_starlark",
            build_setting_default = "starlark_default_value",
        )
        """);
    useConfiguration(
        "--flag_alias=existing_flag=//test:foo_starlark",
        "--existing_flag=cmd_value",
        "--experimental_exec_config=//test:defs.bzl%my_transition");

    getConfiguredTarget("//test:t1");

    var baselineExecConfig = execConfig;
    assertThat(baselineExecConfig.getOptions().get(DummyTestOptions.class).existingFlag)
        .isEqualTo("transitioned native_default_value");
    assertThat(baselineExecConfig.getOptions().getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//test:foo_starlark"), "transitioned cmd_value");

    var t2ExecConfig =
        getConfiguration(Iterables.getOnlyElement(getComputedConfiguredTarget("//test:t2")));
    assertThat(t2ExecConfig.getOptions().get(DummyTestOptions.class).existingFlag)
        .isEqualTo("transitioned native_default_value");
    assertThat(t2ExecConfig.getOptions().getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//test:foo_starlark"), "transitioned cmd_value");

    var t3NonExecConfig =
        getConfiguration(Iterables.getOnlyElement(getComputedConfiguredTarget("//test:t3")));
    assertThat(t3NonExecConfig.getOptions().get(DummyTestOptions.class).existingFlag)
        .isEqualTo("native_default_value");
    assertThat(t3NonExecConfig.getOptions().getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//test:foo_starlark"), "transitioned cmd_value");
  }

  @Test
  public void stampTransitionOutput_stampSettingMarkerNotApplied(@TestParameter boolean stampFlag)
      throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _stamp_output_impl(settings, attr):
            return {"//command_line_option:stamp": True}

        stamp_output_transition = transition(
            implementation = _stamp_output_impl,
            inputs = [],
            outputs = ["//command_line_option:stamp"],
        )

        example = rule(implementation = lambda ctx: None, cfg = stamp_output_transition)
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":defs.bzl", "example")
        example(name = "depends_on_stamp_output")
        """);

    useConfiguration("--stamp=" + stampFlag);

    ActionLookupKey key = getConfiguredTarget("//test:depends_on_stamp_output").getLookupKey();
    NodeEntry node =
        getSkyframeExecutor().getEvaluator().getExistingEntryAtCurrentlyEvaluatingVersion(key);
    assertThat(node.getDirectDeps()).doesNotContain(PrecomputedValue.STAMP_SETTING_MARKER.getKey());
  }

  @Test
  public void stampTransitionInput_stampSettingMarkerAppliedIfStampFlag(
      @TestParameter boolean stampFlag) throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _stamp_input_impl(settings, attr):
            result = "opt" if settings["//command_line_option:stamp"] else "fastbuild"
            return {"//command_line_option:compilation_mode": result}

        stamp_input_transition = transition(
            implementation = _stamp_input_impl,
            inputs = ["//command_line_option:stamp"],
            outputs = ["//command_line_option:compilation_mode"],
        )

        example = rule(implementation = lambda ctx: None, cfg = stamp_input_transition)
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":defs.bzl", "example")
        example(name = "depends_on_stamp_input")
        """);

    useConfiguration("--stamp=" + stampFlag);

    ActionLookupKey key = getConfiguredTarget("//test:depends_on_stamp_input").getLookupKey();
    NodeEntry node =
        getSkyframeExecutor().getEvaluator().getExistingEntryAtCurrentlyEvaluatingVersion(key);
    if (stampFlag) {
      assertThat(node.getDirectDeps()).contains(PrecomputedValue.STAMP_SETTING_MARKER.getKey());
    } else {
      assertThat(node.getDirectDeps())
          .doesNotContain(PrecomputedValue.STAMP_SETTING_MARKER.getKey());
    }
  }

  private ImmutableList<ConfiguredTarget> getComputedConfiguredTarget(String label) {
    return skyframeExecutor.getEvaluator().getDoneValues().entrySet().stream()
        .filter(
            e ->
                e.getKey() instanceof ConfiguredTargetKey ctKey
                    && ctKey.getLabel().toString().equals(label))
        .map(e -> ((ConfiguredTargetValue) e.getValue()).getConfiguredTarget())
        .collect(toImmutableList());
  }
}
