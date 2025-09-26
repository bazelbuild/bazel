// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.select;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link ConfiguredAttributeMapper}.
 *
 * <p>This is distinct from {@link ConfiguredAttributeMapperCommonTest} because the latter needs to
 * inherit from {@link AbstractAttributeMapperTest} to run tests common to all attribute mappers.
 */
@RunWith(JUnit4.class)
public class ConfiguredAttributeMapperTest extends BuildViewTestCase {

  /**
   * Returns a ConfiguredAttributeMapper bound to the given rule with the target configuration.
   */
  private ConfiguredAttributeMapper getMapper(String label) throws Exception {
    ConfiguredTargetAndData ctad = getConfiguredTargetAndData(label);
    return getMapperFromConfiguredTargetAndTarget(ctad);
  }

  private void writeConfigRules() throws Exception {
    scratch.file(
        "conditions/BUILD",
        """
        config_setting(
            name = "a",
            values = {"define": "mode=a"},
        )

        config_setting(
            name = "b",
            values = {"define": "mode=b"},
        )
        """);
  }

  /**
   * Tests that {@link ConfiguredAttributeMapper#get} only gets the configuration-appropriate
   * value.
   */
  @Test
  public void testGetAttribute() throws Exception {
    writeConfigRules();
    scratch.file("a/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    srcs = [],",
        "    outs = ['out'],",
        "    cmd = select({",
        "        '//conditions:a': 'a command',",
        "        '//conditions:b': 'b command',",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': 'default command',",
        "    }))");

    useConfiguration("--define", "mode=a");
    assertThat(getMapper("//a:gen").get("cmd", Type.STRING)).isEqualTo("a command");

    useConfiguration("--define", "mode=b");
    assertThat(getMapper("//a:gen").get("cmd", Type.STRING)).isEqualTo("b command");

    useConfiguration("--define", "mode=c");
    assertThat(getMapper("//a:gen").get("cmd", Type.STRING)).isEqualTo("default command");
  }

  /**
   * Tests that label visitation only travels down configuration-appropriate paths.
   */
  @Test
  public void testLabelVisitation() throws Exception {
    writeConfigRules();
    scratch.file(
        "a/BUILD",
        "load('//test_defs:foo_binary.bzl', 'foo_binary')",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.sh'],",
        "    deps = select({",
        "        '//conditions:a': [':adep'],",
        "        '//conditions:b': [':bdep'],",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': [':defaultdep'],",
        "    }))",
        "foo_library(",
        "    name = 'adep',",
        "    srcs = ['adep.sh'])",
        "foo_library(",
        "    name = 'bdep',",
        "    srcs = ['bdep.sh'])",
        "foo_library(",
        "    name = 'defaultdep',",
        "    srcs = ['defaultdep.sh'])");

    List<Label> visitedLabels = new ArrayList<>();
    Label binSrc = Label.parseCanonical("//a:bin.sh");

    useConfiguration("--define", "mode=a");
    addRelevantLabels(getMapper("//a:bin"), visitedLabels);
    assertThat(visitedLabels).containsExactly(binSrc, Label.parseCanonical("//a:adep"));

    visitedLabels.clear();
    useConfiguration("--define", "mode=b");
    addRelevantLabels(getMapper("//a:bin"), visitedLabels);
    assertThat(visitedLabels).containsExactly(binSrc, Label.parseCanonical("//a:bdep"));

    visitedLabels.clear();
    useConfiguration("--define", "mode=c");
    addRelevantLabels(getMapper("//a:bin"), visitedLabels);
    assertThat(visitedLabels).containsExactly(binSrc, Label.parseCanonical("//a:defaultdep"));
  }

  private static void addRelevantLabels(
      ConfiguredAttributeMapper mapper, List<Label> visitedLabels) {
    mapper.visitAllLabels(
        (attribute, label) -> {
          if (label.getPackageIdentifier().getPackageFragment().toString().equals("a")) {
            visitedLabels.add(label);
          }
        });
  }

  /**
   * Tests that for configurable attributes where the *values* are evaluated in different
   * configurations, the configuration checking still uses the original configuration.
   */
  @Test
  public void testConfigurationTransitions() throws Exception {
    writeConfigRules();
    scratch.file(
        "a/BUILD",
        "load('//test_defs:foo_binary.bzl', 'foo_binary')",
        "genrule(",
        "    name = 'gen',",
        "    srcs = [],",
        "    outs = ['out'],",
        "    cmd = 'nothing',",
        "    tools = select({",
        "        '//conditions:a': [':adep'],",
        "        '//conditions:b': [':bdep'],",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': [':defaultdep'],",
        "    }))",
        "foo_binary(",
        "    name = 'adep',",
        "    srcs = ['adep.sh'])",
        "foo_binary(",
        "    name = 'bdep',",
        "    srcs = ['bdep.sh'])",
        "foo_binary(",
        "    name = 'defaultdep',",
        "    srcs = ['defaultdep.sh'])");
    useConfiguration("--define", "mode=b");

    // Target configuration is in dbg mode, so we should match //conditions:b:
    assertThat(getMapper("//a:gen").get("tools", BuildType.LABEL_LIST))
        .containsExactly(Label.parseCanonical("//a:bdep"));

    // Verify the "tools" dep uses a different configuration that's not also in "dbg":
    assertThat(
            getTarget("//a:gen")
                .getAssociatedRule()
                .getRuleClassObject()
                .getAttributeProvider()
                .getAttributeByName("tools")
                .getTransitionFactory()
                .isTool())
        .isTrue();
    assertThat(getExecConfiguration().getCompilationMode()).isEqualTo(CompilationMode.OPT);
  }

  @Test
  public void testConcatenatedSelects() throws Exception {
    scratch.file(
        "hello/BUILD",
        """
        config_setting(
            name = "a",
            values = {"define": "foo=a"},
        )

        config_setting(
            name = "b",
            values = {"define": "foo=b"},
        )

        config_setting(
            name = "c",
            values = {"define": "bar=c"},
        )

        config_setting(
            name = "d",
            values = {"define": "bar=d"},
        )

        genrule(
            name = "gen",
            srcs = select({
                       ":a": ["a.in"],
                       ":b": ["b.in"],
                   }) +
                   select({
                       ":c": ["c.in"],
                       ":d": ["d.in"],
                   }),
            outs = ["out"],
            cmd = "nothing",
        )
        """);
    useConfiguration("--define", "foo=a", "--define", "bar=d");
    assertThat(getMapper("//hello:gen").get("srcs", BuildType.LABEL_LIST))
        .containsExactly(
            Label.parseCanonical("//hello:a.in"), Label.parseCanonical("//hello:d.in"));
  }

  @Test
  public void testNoneValuesMeansAttributeIsNotExplicitlySet() throws Exception {
    writeConfigRules();
    scratch.file(
        "a/BUILD",
        """
        genrule(
            name = "gen",
            srcs = [],
            outs = ["out"],
            cmd = "",
            message = select({
                "//conditions:a": "defined message",
                "//conditions:b": None,
            }),
        )
        """);

    useConfiguration("--define", "mode=a");
    assertThat(getMapper("//a:gen").isAttributeValueExplicitlySpecified("message")).isTrue();

    useConfiguration("--define", "mode=b");
    assertThat(getMapper("//a:gen").isAttributeValueExplicitlySpecified("message")).isFalse();
  }

  @Test
  public void testNoneValuesWithMultipleSelectsAllNone() throws Exception {
    writeConfigRules();
    scratch.file(
        "a/BUILD",
        """
        genrule(
            name = "gen",
            srcs = [],
            outs = ["out"],
            cmd = "",
            message = select({
                "//conditions:a": "defined message 1",
                "//conditions:b": None,
            }) + select({
                "//conditions:a": "defined message 2",
                "//conditions:b": None,
            }),
        )
        """);

    useConfiguration("--define", "mode=a");
    assertThat(getMapper("//a:gen").isAttributeValueExplicitlySpecified("message")).isTrue();

    useConfiguration("--define", "mode=b");
    assertThat(getMapper("//a:gen").isAttributeValueExplicitlySpecified("message")).isFalse();
  }

  @Test
  public void testNoneValueOnDefaultConditionWithNullDefault() throws Exception {
    writeConfigRules();
    scratch.file(
        "a/BUILD",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.cc'],",
        "    linkstamp = select({",
        "        '//conditions:a': 'notused_linkstamp.cc',",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': None,",
        "    }),",
        ")");

    useConfiguration();
    assertThat(getMapper("//a:lib").isAttributeValueExplicitlySpecified("linkstamp")).isFalse();
    assertThat(getMapper("//a:lib").get("linkstamp", BuildType.LABEL)).isNull();
  }

  @Test
  public void testNoneValueOnMandatoryAttribute() throws Exception {
    scratch.file(
        "a/BUILD",
        """
        # Needed to avoid select() being eliminated as trivial.
        config_setting(
            name = "config",
            values = {"define": "pi=3"},
        )

        alias(
            name = "a",
            actual = select({":config": None, "//conditions:default": None}),
        )
        """);
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("Mandatory attribute 'actual' resolved to 'None'");
  }

  @Test
  public void testAliasedConfigSetting() throws Exception {
    writeConfigRules();
    scratch.file(
        "a/BUILD",
        """
        alias(
            name = "aliased_a",
            actual = "//conditions:a",
        )

        genrule(
            name = "gen",
            srcs = [],
            outs = ["out"],
            cmd = "",
            message = select({
                ":aliased_a": "defined message",
                "//conditions:default": None,
            }),
        )
        """);
    useConfiguration("--define", "mode=a");
    assertThat(getMapper("//a:gen").get("message", Type.STRING)).isEqualTo("defined message");
  }

  @Test
  public void noMatchErrorFormat() throws Exception {
    scratch.file(
        "a/BUILD",
        """
        config_setting(
            name = "a",
            values = {"define": "mode=a"},
        )

        config_setting(
            name = "b",
            values = {"define": "mode=b"},
        )

        genrule(
            name = "g",
            srcs = [],
            outs = ["out"],
            cmd = "",
            message = select({
                ":a": "not chosen",
                ":b": "not chosen",
            }),
        )
        """);
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:g");
    // Match with a regex pattern because the error message includes the failing target's
    // configuration ID, which can vary among builds.
    assertContainsEvent(
        Pattern.compile(
            ".*configurable attribute \"message\" in //a:g doesn't match this configuration. Would"
                + " a default condition help\\?\n"
                + "\n"
                + "Conditions checked:\n"
                + " //a:a\n"
                + " //a:b\n"
                + "\n"
                + "To see a condition's definition, run: bazel query --output=build <condition"
                + " label>.\n"
                + "\n"
                + "This instance of //a:g has configuration identifier .*. To inspect its"
                + " configuration, run: bazel config .*.\n"
                + "\n"
                + "For more help, see"
                + " https://bazel.build/docs/configurable-attributes"
                + "#faq-select-choose-condition.*"));
  }
}
