// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import java.util.Map;
import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that check that dependency cycles are reported correctly. */
@RunWith(JUnit4.class)
public class CircularDependencyTest extends BuildViewTestCase {

  @Test
  public void testOneRuleCycle() throws Exception {
    checkError(
        "cycle",
        "foo.g",
        // error message
        selfEdgeMsg("//cycle:foo.g"),
        // Rule
        "genrule(name = 'foo.g',",
        "        outs = ['Foo.java'],",
        "        srcs = ['foo.g'],",
        "        cmd = 'cat $(SRCS) > $<' )");
  }

  @Test
  public void testDirectPackageGroupCycle() throws Exception {
    checkError(
        "cycle",
        "melon",
        selfEdgeMsg("//cycle:moebius"),
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "package_group(name='moebius', packages=[], includes=['//cycle:moebius'])",
        "foo_library(name='melon', visibility=[':moebius'])");
  }

  @Test
  public void testThreeLongPackageGroupCycle() throws Exception {
    @SuppressWarnings("ConstantPatternCompile")
    Pattern expectedEvent =
        Pattern.compile(
            "cycle in dependency graph:\n"
                + "    //cycle:superman \\([a-f0-9]+\\)\n"
                + ".-> //cycle:rock \\(null\\)\n"
                + "|   //cycle:paper \\(null\\)\n"
                + "|   //cycle:scissors \\(null\\)\n"
                + "`-- //cycle:rock \\(null\\)");
    checkError(
        "cycle",
        "superman",
        expectedEvent,
        "# dummy line",
        "package_group(name='paper', includes=['//cycle:scissors'])",
        "package_group(name='rock', includes=['//cycle:paper'])",
        "package_group(name='scissors', includes=['//cycle:rock'])",
        "filegroup(name='superman', visibility=[':rock'])");

    Event foundEvent = assertContainsEvent(expectedEvent);
    assertThat(foundEvent.getLocation().toString()).isEqualTo("/workspace/cycle/BUILD:3:14");
  }

  /** Test to detect implicit input/output file overlap in rules. */
  @Test
  public void testOneRuleImplicitCycleJava() throws Exception {
    Package pkg =
        createScratchPackageForImplicitCycle(
            "cycle",
            "load('@rules_java//java:defs.bzl', 'java_library')",
            "java_library(name='jcyc',",
            "      srcs = ['libjcyc.jar', 'foo.java'])");
    assertThrows(NoSuchTargetException.class, () -> pkg.getTarget("jcyc"));
    assertThat(pkg.containsErrors()).isTrue();
    assertContainsEvent("rule 'jcyc' has file 'libjcyc.jar' as both an" + " input and an output");
  }

  /**
   * Test not to detect implicit input/output file overlap in rules, when coming from a different
   * package.
   */
  @Test
  public void testInputOutputConflictDifferentPackage() throws Exception {
    Package pkg =
        createScratchPackageForImplicitCycle(
            "googledata/xxx",
            "genrule(name='geo',",
            "    srcs = ['//googledata/geo:geo_info.txt'],",
            "    outs = ['geoinfo.txt'],",
            "    cmd = '$(SRCS) > $@')");
    assertThat(pkg.containsErrors()).isFalse();
  }

  @Test
  public void testTwoRuleCycle() throws Exception {
    scratchRule(
        "b",
        "rule2",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_library(name='rule2',",
        "           deps=['//a:rule1'])");

    checkError(
        "a",
        "rule1",
        Pattern.compile(
            "in cc_library rule //a:rule1: cycle in dependency graph:\n"
                + ".-> //a:rule1 \\([a-f0-9]+\\)\n"
                + "|   //b:rule2 \\([a-f0-9]+\\)\n"
                + "`-- //a:rule1 \\([a-f0-9]+\\)"),
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_library(name='rule1',",
        "           deps=['//b:rule2'])");
  }

  @Test
  public void testTwoRuleCycle2() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file(
        "x/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "x",
            deps = ["y"],
        )

        java_library(
            name = "y",
            deps = ["x"],
        )
        """);
    getConfiguredTarget("//x");
    assertContainsEvent("in java_library rule //x:x: cycle in dependency graph");
  }

  @Test
  public void testIndirectOneRuleCycle() throws Exception {
    scratchRule(
        "cycle",
        "foo.h",
        "genrule(name = 'foo.h',",
        "      outs = ['bar.h'],",
        "      srcs = ['foo.h'],",
        "      cmd = 'cp $< $@')");
    checkError(
        "main",
        "mygenrule",
        // error message
        selfEdgeMsg("//cycle:foo.h"),
        // Rule
        "genrule(name='mygenrule',",
        "      outs = ['baz.h'],",
        "      srcs = ['//cycle:foo.h'],",
        "      cmd = 'cp $< $@')");
  }

  private Pattern selfEdgeMsg(String label) {
    return Pattern.compile(label + " \\([a-f0-9]+|null\\) \\[self-edge\\]");
  }

  // Regression test for: "IllegalStateException in
  // AbstractConfiguredTarget.initialize()".
  // Failure to mark all cycle-forming nodes when there are *two* cycles led to
  // an attempt to initialise a node we'd already visited.
  @Test
  public void testTwoCycles() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file(
        "x/BUILD",
        """
        genrule(
            name = "b",
            srcs = ["c"],
            outs = ["b.out"],
            cmd = ":",
            tools = ["c"],
        )

        genrule(
            name = "c",
            srcs = ["b.out"],
            outs = [],
            cmd = ":",
        )
        """);
    getConfiguredTarget("//x:b"); // doesn't crash!
    assertContainsEvent("cycle in dependency graph");
  }

  @Test
  public void testAspectCycle() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "x/BUILD",
        """
        load("//x:x.bzl", "aspected", "plain")

        # Using data= makes the dependency graph clearer because then the aspect does not propagate
        # from aspectdep through a to b (and c)
        plain(
            name = "a",
            noaspect_deps = [":b"],
        )

        aspected(
            name = "b",
            aspect_deps = ["c"],
        )

        plain(name = "c")

        plain(
            name = "aspectdep",
            aspect_deps = ["a"],
        )
        """);

    scratch.file(
        "x/x.bzl",
        """
        def _impl(ctx):
            return []

        rule_aspect = aspect(
            implementation = _impl,
            attr_aspects = ["aspect_deps"],
            attrs = {"_implicit": attr.label(default = Label("//x:aspectdep"))},
        )

        plain = rule(
            implementation = _impl,
            attrs = {"aspect_deps": attr.label_list(), "noaspect_deps": attr.label_list()},
        )

        aspected = rule(
            implementation = _impl,
            attrs = {"aspect_deps": attr.label_list(aspects = [rule_aspect])},
        )
        """);

    getConfiguredTarget("//x:a");
    assertContainsEvent("cycle in dependency graph");
    assertContainsEvent("//x:c with aspect //x:x.bzl%rule_aspect");
  }

  /** A late bound dependency which depends on the 'dep' label if the 'define' is in --defines. */
  // TODO(b/65746853): provide a way to do this without passing the entire configuration
  private static final LabelLateBoundDefault<BuildConfigurationValue> LATE_BOUND_DEP =
      LabelLateBoundDefault.fromTargetConfiguration(
          BuildConfigurationValue.class,
          null,
          (rule, attributes, config) ->
              config.getCommandLineBuildVariables().containsKey(attributes.get("define", STRING))
                  ? attributes.get("dep", NODEP_LABEL)
                  : null);

  /** A rule which always depends on the given label. */
  private static final MockRule NORMAL_DEPENDER =
      () -> MockRule.define("normal_dep", attr("dep", LABEL).allowedFileTypes());

  /** A rule which depends on a given label only if the given define is set. */
  private static final MockRule LATE_BOUND_DEPENDER =
      () ->
          MockRule.define(
              "late_bound_dep",
              attr("define", STRING).mandatory(),
              attr("dep", NODEP_LABEL).mandatory(),
              attr(":late_bound_dep", LABEL).value(LATE_BOUND_DEP));

  /** A rule which removes a define from the configuration of its dependency. */
  private static final MockRule DEFINE_CLEARER =
      () ->
          MockRule.define(
              "define_clearer",
              attr("define", STRING).mandatory(),
              attr("dep", LABEL)
                  .mandatory()
                  .allowedFileTypes()
                  .cfg(
                      new TransitionFactory<>() {
                        @Override
                        public SplitTransition create(AttributeTransitionData data) {
                          return new SplitTransition() {

                            @Override
                            public ImmutableSet<Class<? extends FragmentOptions>>
                                requiresOptionFragments() {
                              return ImmutableSet.of(CoreOptions.class);
                            }

                            @Override
                            public Map<String, BuildOptions> split(
                                BuildOptionsView options, EventHandler eventHandler) {
                              String define = data.attributes().get("define", STRING);
                              BuildOptionsView newOptions = options.clone();
                              CoreOptions optionsFragment = newOptions.get(CoreOptions.class);
                              optionsFragment.commandLineBuildVariables =
                                  optionsFragment.commandLineBuildVariables.stream()
                                      .filter((pair) -> !pair.getKey().equals(define))
                                      .collect(toImmutableList());
                              return ImmutableMap.of("define_cleaner", newOptions.underlying());
                            }
                          };
                        }

                        @Override
                        public TransitionType transitionType() {
                          return TransitionType.ATTRIBUTE;
                        }

                        @Override
                        public boolean isSplit() {
                          return true;
                        }
                      }));

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder()
            .addRuleDefinition(NORMAL_DEPENDER)
            .addRuleDefinition(LATE_BOUND_DEPENDER)
            .addRuleDefinition(DEFINE_CLEARER);
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Test
  public void testLateBoundTargetCycleNotConfiguredTargetCycle() throws Exception {
    // Target graph: //a -> //b -?> //c -> //a (loop)
    // Configured target graph: //a -> //b -> //c -> //a (2) -> //b (2)
    scratch.file("a/BUILD", "normal_dep(name = 'a', dep = '//b')");
    scratch.file("b/BUILD", "late_bound_dep(name = 'b', dep = '//c', define = 'CYCLE_ON')");
    scratch.file("c/BUILD", "define_clearer(name = 'c', dep = '//a', define = 'CYCLE_ON')");

    useConfiguration("--define=CYCLE_ON=yes");
    getConfiguredTarget("//a");
    assertNoEvents();
  }

  @Test
  public void testSelectTargetCycleNotConfiguredTargetCycle() throws Exception {
    // Target graph: //a -> //b -?> //c -> //a (loop)
    // Configured target graph: //a -> //b -> //c -> //a (2) -> //b (2) -> //b:stop (2)
    scratch.file("a/BUILD", "normal_dep(name = 'a', dep = '//b')");
    scratch.file(
        "b/BUILD",
        """
        config_setting(
            name = "cycle",
            define_values = {"CYCLE_ON": "yes"},
        )

        normal_dep(name = "stop")

        normal_dep(
            name = "b",
            dep = select({
                ":cycle": "//c",
                "//conditions:default": ":stop",
            }),
        )
        """);
    scratch.file("c/BUILD", "define_clearer(name = 'c', dep = '//a', define = 'CYCLE_ON')");

    useConfiguration("--define=CYCLE_ON=yes");
    getConfiguredTarget("//a");
    assertNoEvents();
  }

  @Test
  public void testInvalidVisibility() throws Exception {
    scratch.file(
        "a/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "rule1",
            visibility = ["//b:rule2"],
            deps = ["//b:rule2"],
        )
        """);
    scratch.file(
        "b/BUILD",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_library(name='rule2')");

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//a:rule1"));

    assertThat(expected)
        .hasMessageThat()
        .contains("Label '//b:rule2' does not refer to a package group.");
  }

  @Test
  public void testInvalidVisibilityWithSelect() throws Exception {
    scratch.file(
        "a/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "rule1",
            visibility = ["//b:rule2"],
            deps = ["//b:rule2"],
        )
        """);
    scratch.file(
        "b/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        config_setting(
            name = "fastbuild",
            values = {"compilation_mode": "fastbuild"},
        )

        cc_library(
            name = "rule2",
            hdrs = select({
                ":fastbuild": glob(
                    [
                        "*.h",
                    ],
                    allow_empty = True,
                ),
            }),
        )
        """);

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//a:rule1"));

    assertThat(expected)
        .hasMessageThat()
        .contains("Label '//b:rule2' does not refer to a package group.");
  }
}
