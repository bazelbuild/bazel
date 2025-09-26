// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlark;

import static com.google.common.truth.Truth.assertThat;
import static java.util.stream.Collectors.joining;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.pkgcache.TargetParsingCompleteEvent;
import com.google.devtools.build.lib.starlark.util.StarlarkOptionsTestCase;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import net.starlark.java.eval.StarlarkInt;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit test for the {@code StarlarkOptionsParser}. */
@RunWith(TestParameterInjector.class)
public class StarlarkOptionsParsingTest extends StarlarkOptionsTestCase {

  private List<Postable> postedEvents;

  @Before
  public void addPostableEventHandler() {
    postedEvents = new ArrayList<>();
    reporter.addHandler(
        new ExtendedEventHandler() {
          @Override
          public void post(Postable obj) {
            postedEvents.add(obj);
          }

          @Override
          public void handle(Event event) {}
        });
  }

  /** Returns only the posted events of the given class. */
  private List<Postable> eventsOfType(Class<? extends Postable> clazz) {
    return postedEvents.stream()
        .filter(event -> event.getClass().equals(clazz))
        .collect(Collectors.toList());
  }

  // test --flag=value
  @Test
  public void testFlagEqualsValueForm() throws Exception {
    writeBasicIntFlag();

    OptionsParsingResult result = parseStarlarkOptions("--//test:my_int_setting=666");

    assertThat(result.getStarlarkOptions()).hasSize(1);
    assertThat(result.getStarlarkOptions().get("//test:my_int_setting"))
        .isEqualTo(StarlarkInt.of(666));
    assertThat(result.getResidue()).isEmpty();
  }

  // test --@main_workspace//flag=value parses out to //flag=value
  // test --@other_workspace//flag=value parses out to @other_workspace//flag=value
  @Test
  public void testFlagNameWithExternalRepo() throws Exception {
    writeBasicIntFlag();
    scratch.file("test/repo2/MODULE.bazel", "module(name = 'repo2')");
    scratch.file(
        "test/repo2/defs.bzl",
        """
        def _impl(ctx):
            pass

        my_flag = rule(
            implementation = _impl,
            build_setting = config.int(flag = True),
        )
        """);
    scratch.file(
        "test/repo2/BUILD",
        """
        load(":defs.bzl", "my_flag")

        my_flag(
            name = "flag2",
            build_setting_default = 2,
        )
        """);

    rewriteModuleDotBazel(
        """
        module(name = "starlark_options_test")

        bazel_dep(name = "repo2")

        local_path_override(
            module_name = "repo2",
            path = "test/repo2",
        )
        """);

    OptionsParsingResult result =
        parseStarlarkOptions(
            "--@starlark_options_test//test:my_int_setting=666 --@repo2//:flag2=222",
            /* onlyStarlarkParser= */ true);

    assertThat(result.getStarlarkOptions()).hasSize(2);
    assertThat(result.getStarlarkOptions().get("//test:my_int_setting"))
        .isEqualTo(StarlarkInt.of(666));
    assertThat(result.getStarlarkOptions().get("@@repo2+//:flag2")).isEqualTo(StarlarkInt.of(222));
    assertThat(result.getResidue()).isEmpty();
  }

  // test --fake_flag=value
  @Test
  public void testBadFlag_equalsForm() throws Exception {
    scratch.file("test/BUILD");
    reporter.removeHandler(failFastHandler);

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> parseStarlarkOptions("--//fake_flag=blahblahblah"));

    assertThat(e).hasMessageThat().contains("Error loading option //fake_flag");
    assertThat(e.getInvalidArgument()).isEqualTo("//fake_flag");
  }

  // test --fake_flag
  @Test
  public void testBadFlag_boolForm() throws Exception {
    scratch.file("test/BUILD");
    reporter.removeHandler(failFastHandler);

    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> parseStarlarkOptions("--//fake_flag"));

    assertThat(e).hasMessageThat().contains("Error loading option //fake_flag");
    assertThat(e.getInvalidArgument()).isEqualTo("//fake_flag");
  }

  @Test
  public void testBadFlag_keepGoing() throws Exception {
    optionsParser.parse("--keep_going");
    scratch.file("test/BUILD");
    reporter.removeHandler(failFastHandler);

    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> parseStarlarkOptions("--//fake_flag"));

    assertThat(e).hasMessageThat().contains("Error loading option //fake_flag");
    assertThat(e.getInvalidArgument()).isEqualTo("//fake_flag");
  }

  @Test
  public void testSingleDash_notAllowed() throws Exception {
    writeBasicIntFlag();

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () ->
                parseStarlarkOptions("-//test:my_int_setting=666", /* onlyStarlarkParser= */ true));
    assertThat(e).hasMessageThat().isEqualTo("Invalid options syntax: -//test:my_int_setting=666");
  }

  // test --non_flag_setting=value
  @Test
  public void testNonFlagParsing() throws Exception {
    scratch.file(
        "test/build_setting.bzl",
        """
        def _build_setting_impl(ctx):
            return []

        int_flag = rule(
            implementation = _build_setting_impl,
            build_setting = config.int(flag = False),
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:build_setting.bzl", "int_flag")

        int_flag(
            name = "my_int_setting",
            build_setting_default = 42,
        )
        """);

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> parseStarlarkOptions("--//test:my_int_setting=666"));

    assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: //test:my_int_setting=666");
  }

  // test --bool_flag
  @Test
  public void testBooleanFlag() throws Exception {
    writeBasicBoolFlag();

    OptionsParsingResult result = parseStarlarkOptions("--//test:my_bool_setting=false");

    assertThat(result.getStarlarkOptions()).hasSize(1);
    assertThat(result.getStarlarkOptions().get("//test:my_bool_setting")).isEqualTo(false);
    assertThat(result.getResidue()).isEmpty();
  }

  // test --nobool_flag
  @Test
  public void testNoPrefixedBooleanFlag() throws Exception {
    writeBasicBoolFlag();

    OptionsParsingResult result = parseStarlarkOptions("--no//test:my_bool_setting");

    assertThat(result.getStarlarkOptions()).hasSize(1);
    assertThat(result.getStarlarkOptions().get("//test:my_bool_setting")).isEqualTo(false);
    assertThat(result.getResidue()).isEmpty();
  }

  // test --no@main_workspace//:bool_flag
  @Test
  public void testNoPrefixedBooleanFlag_withWorkspace() throws Exception {
    writeBasicBoolFlag();

    OptionsParsingResult result = parseStarlarkOptions("--no@//test:my_bool_setting");

    assertThat(result.getStarlarkOptions()).hasSize(1);
    assertThat(result.getStarlarkOptions().get("//test:my_bool_setting")).isEqualTo(false);
    assertThat(result.getResidue()).isEmpty();
  }

  // test --noint_flag
  @Test
  public void testNoPrefixedNonBooleanFlag() throws Exception {
    writeBasicIntFlag();

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class, () -> parseStarlarkOptions("--no//test:my_int_setting"));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo("Illegal use of 'no' prefix on non-boolean option: //test:my_int_setting");
  }

  // test --int_flag
  @Test
  public void testFlagWithoutValue() throws Exception {
    writeBasicIntFlag();

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class, () -> parseStarlarkOptions("--//test:my_int_setting"));

    assertThat(e).hasMessageThat().isEqualTo("Expected value after --//test:my_int_setting");
  }

  // test --flag --flag
  @Test
  public void testRepeatFlagLastOneWins() throws Exception {
    writeBasicIntFlag();

    OptionsParsingResult result =
        parseStarlarkOptions("--//test:my_int_setting=4 --//test:my_int_setting=7");

    assertThat(result.getStarlarkOptions()).hasSize(1);
    assertThat(result.getStarlarkOptions().get("//test:my_int_setting"))
        .isEqualTo(StarlarkInt.of(7));
    assertThat(result.getResidue()).isEmpty();
  }

  // test --flagA=valueA --flagB=valueB
  @Test
  public void testMultipleFlags() throws Exception {
    scratch.file(
        "test/build_setting.bzl",
        """
        def _build_setting_impl(ctx):
            return []

        int_flag = rule(
            implementation = _build_setting_impl,
            build_setting = config.int(flag = True),
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:build_setting.bzl", "int_flag")

        int_flag(
            name = "my_int_setting",
            build_setting_default = 42,
        )

        int_flag(
            name = "my_other_int_setting",
            build_setting_default = 77,
        )
        """);

    OptionsParsingResult result =
        parseStarlarkOptions("--//test:my_int_setting=0 --//test:my_other_int_setting=0");

    assertThat(result.getResidue()).isEmpty();
    assertThat(result.getStarlarkOptions()).hasSize(2);
    assertThat(result.getStarlarkOptions().get("//test:my_int_setting"))
        .isEqualTo(StarlarkInt.of(0));
    assertThat(result.getStarlarkOptions().get("//test:my_other_int_setting"))
        .isEqualTo(StarlarkInt.of(0));
  }

  // test --non_build_setting
  @Test
  public void testNonBuildSetting() throws Exception {
    scratch.file(
        "test/rules.bzl",
        """
        def _impl(ctx):
            return []

        my_rule = rule(
            implementation = _impl,
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:rules.bzl", "my_rule")

        my_rule(name = "my_rule")
        """);
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> parseStarlarkOptions("--//test:my_rule"));
    assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: //test:my_rule");
  }

  // test --non_rule_configured_target
  @Test
  public void testNonRuleConfiguredTarget() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        genrule(
            name = "my_gen",
            srcs = ["x.in"],
            outs = ["x.cc"],
            cmd = "$(locations :tool) $< >$@",
            tools = [":tool"],
        )

        filegroup(name = "tool-dep")
        """);
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> parseStarlarkOptions("--//test:x.in"));
    assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: //test:x.in");
  }

  // test --int_flag=non_int_value
  @Test
  public void testWrongValueType_int() throws Exception {
    writeBasicIntFlag();

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> parseStarlarkOptions("--//test:my_int_setting=woohoo"));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo("While parsing option //test:my_int_setting=woohoo: 'woohoo' is not a int");
  }

  // test --bool_flag=non_bool_value
  @Test
  public void testWrongValueType_bool() throws Exception {
    writeBasicBoolFlag();

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> parseStarlarkOptions("--//test:my_bool_setting=woohoo"));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo("While parsing option //test:my_bool_setting=woohoo: 'woohoo' is not a boolean");
  }

  // test --int-flag=same value as default
  @Test
  public void testDontStoreDefaultValue() throws Exception {
    // build_setting_default = 42
    writeBasicIntFlag();

    OptionsParsingResult result = parseStarlarkOptions("--//test:my_int_setting=42");

    assertThat(result.getStarlarkOptions()).isEmpty();
  }

  @Test
  public void testOptionsAreParsedWithBuildTestsOnly() throws Exception {
    writeBasicIntFlag();
    optionsParser.parse("--build_tests_only");

    OptionsParsingResult result = parseStarlarkOptions("--//test:my_int_setting=15");

    assertThat(result.getStarlarkOptions().get("//test:my_int_setting"))
        .isEqualTo(StarlarkInt.of(15));
  }

  /**
   * When Starlark flags are only set as flags, they shouldn't produce {@link
   * TargetParsingCompleteEvent}s. That's intended to communicate (to the build event protocol)
   * which of the targets in {@code blaze build //foo:all //bar:all} were built.
   */
  @Test
  public void testExpectedBuildEventOutput_asFlag() throws Exception {
    writeBasicIntFlag();
    scratch.file(
        "blah/BUILD",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_library(name = 'mylib')");
    useConfiguration("--//test:my_int_setting=15");
    update(
        ImmutableList.of("//blah:mylib"),
        /*keepGoing=*/ false,
        /*loadingPhaseThreads=*/ LOADING_PHASE_THREADS,
        /*doAnalysis*/ true,
        eventBus);
    List<Postable> targetParsingCompleteEvents = eventsOfType(TargetParsingCompleteEvent.class);
    assertThat(targetParsingCompleteEvents).hasSize(1);
    assertThat(
            ((TargetParsingCompleteEvent) targetParsingCompleteEvents.get(0))
                .getOriginalTargetPattern())
        .containsExactly("//blah:mylib");
  }

  /**
   * But Starlark are also targets. When they're requested as normal build targets they should
   * produce {@link TargetParsingCompleteEvent} just like any other target.
   */
  @Test
  public void testExpectedBuildEventOutput_asTarget() throws Exception {
    writeBasicIntFlag();
    scratch.file(
        "blah/BUILD",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_library(name = 'mylib')");
    useConfiguration("--//test:my_int_setting=15");
    update(
        ImmutableList.of("//blah:mylib", "//test:my_int_setting"),
        /*keepGoing=*/ false,
        /*loadingPhaseThreads=*/ LOADING_PHASE_THREADS,
        /*doAnalysis*/ true,
        eventBus);
    List<Postable> targetParsingCompleteEvents = eventsOfType(TargetParsingCompleteEvent.class);
    assertThat(targetParsingCompleteEvents).hasSize(1);
    assertThat(
            ((TargetParsingCompleteEvent) targetParsingCompleteEvents.get(0))
                .getOriginalTargetPattern())
        .containsExactly("//blah:mylib", "//test:my_int_setting");
  }

  @Test
  @SuppressWarnings("unchecked")
  public void testAllowMultipleStringFlag() throws Exception {
    scratch.file(
        "test/build_setting.bzl",
        """
        def _build_setting_impl(ctx):
            return []

        allow_multiple_flag = rule(
            implementation = _build_setting_impl,
            build_setting = config.string(flag = True, allow_multiple = True),
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:build_setting.bzl", "allow_multiple_flag")

        allow_multiple_flag(
            name = "cats",
            build_setting_default = "tabby",
        )
        """);

    OptionsParsingResult result = parseStarlarkOptions("--//test:cats=calico --//test:cats=bengal");

    assertThat(result.getStarlarkOptions().keySet()).containsExactly("//test:cats");
    assertThat((List<String>) result.getStarlarkOptions().get("//test:cats"))
        .containsExactly("calico", "bengal");
  }

  @Test
  @SuppressWarnings("unchecked")
  public void testRepeatedStringListFlag() throws Exception {
    scratch.file(
        "test/build_setting.bzl",
        """
        def _build_setting_impl(ctx):
            return []

        repeated_flag = rule(
            implementation = _build_setting_impl,
            build_setting = config.string_list(flag = True, repeatable = True),
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:build_setting.bzl", "repeated_flag")

        repeated_flag(
            name = "cats",
            build_setting_default = ["tabby"],
        )
        """);

    OptionsParsingResult result = parseStarlarkOptions("--//test:cats=calico --//test:cats=bengal");

    assertThat(result.getStarlarkOptions().keySet()).containsExactly("//test:cats");
    assertThat((List<String>) result.getStarlarkOptions().get("//test:cats"))
        .containsExactly("calico", "bengal");
  }

  @Test
  public void flagReferencesExactlyOneTarget() throws Exception {
    scratch.file(
        "test/build_setting.bzl",
        """
        string_flag = rule(
            implementation = lambda ctx, attr: [],
            build_setting = config.string(flag = True),
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:build_setting.bzl", "string_flag")

        string_flag(
            name = "one",
            build_setting_default = "",
        )

        string_flag(
            name = "two",
            build_setting_default = "",
        )
        """);

    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> parseStarlarkOptions("--//test:all"));

    assertThat(e)
        .hasMessageThat()
        .contains("//test:all: user-defined flags must reference exactly one target");
  }

  @Test
  public void flagIsAlias() throws Exception {
    scratch.file(
        "test/build_setting.bzl",
        """
        string_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.string(flag = True),
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:build_setting.bzl", "string_flag")

        alias(
            name = "one",
            actual = "//test/pkg:two",
        )

        string_flag(
            name = "three",
            build_setting_default = "",
        )
        """);
    scratch.file(
        "test/pkg/BUILD",
        """
        alias(
            name = "two",
            actual = "//test:three",
        )
        """);

    OptionsParsingResult result = parseStarlarkOptions("--//test:one=one --//test/pkg:two=two");

    assertThat(result.getStarlarkOptions()).containsExactly("//test:three", "two");
  }

  @Test
  public void flagIsAlias_cycle() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        alias(
            name = "one",
            actual = "//test/pkg:two",
        )

        alias(
            name = "three",
            actual = ":one",
        )
        """);
    scratch.file(
        "test/pkg/BUILD",
        """
        alias(
            name = "two",
            actual = "//test:three",
        )
        """);

    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> parseStarlarkOptions("--//test:one=one"));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Failed to load build setting '//test:one' due to a cycle in alias chain: //test:one"
                + " -> //test/pkg:two -> //test:three -> //test:one");
  }

  @Test
  public void flagIsAlias_usesSelect() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        alias(
            name = "one",
            actual = "//test/pkg:two",
        )

        alias(
            name = "three",
            actual = ":one",
        )
        """);
    scratch.file(
        "test/pkg/BUILD",
        """
        # Needed to avoid select() being eliminated as trivial.
        config_setting(
            name = "config",
            values = {"define": "pi=3"},
        )

        alias(
            name = "two",
            actual = select({
                ":config": "//test:three",
                "//conditions:default": "//test:three",
            }),
        )
        """);

    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> parseStarlarkOptions("--//test:one=one"));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Failed to load build setting '//test:one' as it resolves to an alias with an actual"
                + " value that uses select(): //test:one -> //test/pkg:two. This is not supported"
                + " as build settings are needed to determine the configuration the select is"
                + " evaluated in.");
  }

  @Test
  public void flagIsAlias_resolvesToNonBuildSettingTarget() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        alias(
            name = "one",
            actual = "//test/pkg:two",
        )

        genrule(
            name = "three",
            outs = ["out"],
            cmd = "echo hello > $@",
        )
        """);
    scratch.file(
        "test/pkg/BUILD",
        """
        alias(
            name = "two",
            actual = "//test:three",
        )
        """);

    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> parseStarlarkOptions("--//test:one=one"));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo("Unrecognized option: //test:one -> //test/pkg:two -> //test:three");
  }

  @Test
  @TestParameters({
    // Not repeatable, flag value is not the same as the default, flag added to options.
    "{defaultValue: ['default'], repeatable: false, cmdValue:"
        + " ['v1,v4,v3,v2,v3,v1,v4,v1'], expectedValue: ['v1', 'v2', 'v3', 'v4']}",
    // Repeatable, flag value is not the same as the default, flag added to options.
    "{defaultValue: ['default'], repeatable: true, cmdValue: ['v2', 'v2', 'v1', 'v3', 'v4', 'v1'],"
        + " expectedValue: ['v1', 'v2', 'v3', 'v4']}",
    // Not repeatable, flag value is the same as the default, flag not added to options.
    "{defaultValue: ['v2','v1','v1', 'v3', 'v2', 'v3', 'v4'], repeatable: false, cmdValue:"
        + " ['v1,v4,v3,v2,v3,v1,v4,v1'], expectedValue: null}",
    // Repeatable, flag value is the same as the default, flag not added to options.
    "{defaultValue: ['v2','v1','v1', 'v3', 'v2', 'v3', 'v4'], repeatable: true, cmdValue: ['v2',"
        + " 'v2', 'v1', 'v3', 'v4', 'v1'], expectedValue: null}"
  })
  public void testStringSetFlag(
      List<String> defaultValue,
      boolean repeatable,
      List<String> cmdValue,
      List<String> expectedValue)
      throws Exception {
    scratch.file(
        "test/build_setting.bzl",
        String.format(
            """
            def _build_setting_impl(ctx):
                return []

            string_set_flag = rule(
                implementation = _build_setting_impl,
                build_setting = config.string_set(flag = True, repeatable = %s),
            )
            """,
            repeatable ? "True" : "False"));
    scratch.file(
        "test/BUILD",
        String.format(
            """
            load("//test:build_setting.bzl", "string_set_flag")

            string_set_flag(
                name = "my_flag",
                build_setting_default = set([%s]),
            )
            """,
            defaultValue.stream().map(v -> String.format("'%s'", v)).collect(joining(","))));

    OptionsParsingResult result =
        parseStarlarkOptions(
            cmdValue.stream()
                .map(v -> String.format("--//test:my_flag=%s", v))
                .collect(joining(" ")));

    if (expectedValue == null) {
      assertThat(result.getStarlarkOptions()).isEmpty();
    } else {
      assertThat(result.getStarlarkOptions().keySet()).containsExactly("//test:my_flag");
      assertThat(result.getStarlarkOptions().get("//test:my_flag"))
          .isEqualTo(ImmutableSet.copyOf(expectedValue));
    }
  }
}
