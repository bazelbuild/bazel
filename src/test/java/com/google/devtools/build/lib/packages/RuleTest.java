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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.util.TargetDataSubject.assertThat;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.testutils.RoundTripping;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Rule}. */
@RunWith(JUnit4.class)
public class RuleTest extends PackageLoadingTestCase {
  private static final RuleDefinition FAKE_CC_LIBRARY =
      (MockRule) () -> MockRule.define("fake_cc_library", (builder, env) -> {});

  private static final RuleDefinition FAKE_CC_BINARY =
      (MockRule)
          () ->
              MockRule.define(
                  "fake_cc_binary",
                  (builder, env) -> builder.add(attr("srcs", LABEL_LIST).legacyAllowAnyFileType()));

  private static final RuleDefinition FAKE_CC_TEST =
      (MockRule)
          () ->
              MockRule.ancestor(BaseRuleClasses.NativeBuildRule.class)
                  .type(RuleClassType.TEST)
                  .define(
                      "fake_cc_test",
                      (builder, env) ->
                          builder
                              .add(attr("srcs", LABEL_LIST).legacyAllowAnyFileType())
                              .add(attr("deps", LABEL_LIST).legacyAllowAnyFileType())
                              .add(attr("size", STRING).nonconfigurable("policy").value("small"))
                              .add(attr("timeout", STRING).nonconfigurable("policy").value("short"))
                              .add(attr("flaky", BOOLEAN))
                              .add(attr("shard_count", INTEGER))
                              .add(attr("local", BOOLEAN).nonconfigurable("policy")));

  @Override
  protected ImmutableList<RuleDefinition> getExtraRules() {
    return ImmutableList.of(FAKE_CC_LIBRARY, FAKE_CC_BINARY, FAKE_CC_TEST);
  }

  @Test
  public void testOutputNameError() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "namecollide/BUILD",
        """
        genrule(
            name = "hello_world",
            srcs = ["ignore_me.txt"],
            outs = [
                "message.txt",
                "hello_world",
            ],
            cmd = 'echo "Hello, world." >$(location message.txt)',
        )
        """);
    Rule genRule = (Rule) getTarget("//namecollide:hello_world");
    assertThat(genRule.containsErrors()).isFalse(); // TODO: assertTrue
    assertContainsEvent(
        "target 'hello_world' is both a rule and a file; please choose another name for the rule",
        ImmutableSet.of(EventKind.WARNING));
  }

  @Test
  public void testIsLocalTestRuleForLocalEquals1() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        fake_cc_test(
            name = "y",
            srcs = ["a"],
            local = 0,
        )

        fake_cc_test(
            name = "z",
            srcs = ["a"],
            local = 1,
        )
        """);
    Rule y = (Rule) getTarget("//x:y");
    assertThat(TargetUtils.isLocalTestRule(y)).isFalse();
    Rule z = (Rule) getTarget("//x:z");
    assertThat(TargetUtils.isLocalTestRule(z)).isTrue();
  }

  @Test
  public void testDeprecation() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        fake_cc_test(name = "y")

        fake_cc_test(
            name = "z",
            deprecation = "Foo",
        )
        """);
    Rule y = (Rule) getTarget("//x:y");
    assertThat(TargetUtils.getDeprecation(y)).isNull();
    Rule z = (Rule) getTarget("//x:z");
    assertThat(TargetUtils.getDeprecation(z)).isEqualTo("Foo");
  }

  @Test
  public void testVisibilityValid() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        fake_cc_binary(
            name = "pr",
            visibility = ["//visibility:private"],
        )

        fake_cc_binary(
            name = "pu",
            visibility = ["//visibility:public"],
        )

        fake_cc_binary(
            name = "cu",
            visibility = ["//a:b"],
        )
        """);
    Package pkg = getPackage("x");
    assertThat(pkg.getRule("pu").getVisibility()).isEqualTo(RuleVisibility.PUBLIC);
    assertThat(pkg.getRule("pr").getVisibility()).isEqualTo(RuleVisibility.PRIVATE);
  }

  @Test
  public void testVisibilityTypo_failsCleanly() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        fake_cc_binary(
            name = "typo",
            visibility = ["//visibility:none"],
        )
        """);
    reporter.removeHandler(failFastHandler);
    Package pkg = getPackage("x");
    assertContainsEvent(
        "Invalid visibility label '//visibility:none'; did you mean //visibility:public or"
            + " //visibility:private?");
    assertThat(pkg.containsErrors()).isTrue();
  }

  @Test
  public void testVisibilityTypo_whenVisibilityPackageExists_failsCleanly() throws Exception {
    scratch.file(
        "visibility/BUILD",
        """
        fake_cc_binary(
            name = "none",
        )
        """);
    scratch.file(
        "x/BUILD",
        """
        fake_cc_binary(
            name = "typo",
            visibility = ["//visibility:none"],
        )
        """);
    assertThat(getPackage("visibility").containsErrors()).isFalse();
    reporter.removeHandler(failFastHandler);
    Package pkg = getPackage("x");
    assertContainsEvent(
        "Invalid visibility label '//visibility:none'; did you mean //visibility:public or"
            + " //visibility:private?");
    assertThat(pkg.containsErrors()).isTrue();
  }

  @Test
  public void testVisibilityPkgSubpackages_whenVisibilityPackageExists_succeeds() throws Exception {
    scratch.file(
        "visibility/BUILD",
        """
        fake_cc_binary(
            name = "none",
        )
        """);
    scratch.file(
        "x/BUILD",
        """
        fake_cc_binary(
            name = "p",
            visibility = ["//visibility:__pkg__"],
        )

        fake_cc_binary(
            name = "s",
            visibility = ["//visibility:__subpackages__"],
        )
        """);
    assertThat(getPackage("visibility").containsErrors()).isFalse();
    Package pkg = getPackage("x");
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getRule("p").getVisibility().getDeclaredLabels())
        .containsExactly(Label.parseCanonicalUnchecked("//visibility:__pkg__"));
    assertThat(pkg.getRule("s").getVisibility().getDeclaredLabels())
        .containsExactly(Label.parseCanonicalUnchecked("//visibility:__subpackages__"));
  }

  @Test
  public void testVisibilityMisspelling() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        fake_cc_binary(
            name = "is_this_public",
            visibility = ["//visibility:plubic"],
        )
        """);
    reporter.removeHandler(failFastHandler);
    Package pkg = getPackage("x");
    assertContainsEvent(
        "Invalid visibility label '//visibility:plubic'; did you mean //visibility:public or"
            + " //visibility:private?");
    assertThat(pkg.containsErrors()).isTrue();
  }

  @Test
  public void testPublicAndPrivateVisibility() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        package(default_visibility = ["//default:__pkg__"])

        fake_cc_binary(
            name = "is_this_public",
            visibility = ["//some:__pkg__", "//visibility:public"],
        )

        fake_cc_binary(
            name = "is_private_dropped",
            visibility = ["//some:__pkg__", "//visibility:private"],
        )

        fake_cc_binary(
            name = "is_empty_visibility_private",
            visibility = [],
        )
        """);
    Package pkg = getPackage("x");
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getRule("is_this_public").getVisibility().getDeclaredLabels())
        .containsExactly(Label.parseCanonicalUnchecked("//visibility:public"));
    assertThat(pkg.getRule("is_private_dropped").getVisibility().getDeclaredLabels())
        .containsExactly(Label.parseCanonicalUnchecked("//some:__pkg__"));
    assertThat(pkg.getRule("is_empty_visibility_private").getVisibility().getDeclaredLabels())
        .containsExactly(Label.parseCanonicalUnchecked("//visibility:private"));
  }

  @Test
  public void testReduceForSerialization() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        fake_cc_library(
            name = "dep",
            deprecation = "message should serialize",
        )

        fake_cc_test(
            name = "y",
            srcs = ["a"],
            deps = [":dep"],
        )

        fake_cc_binary(
            name = "cu",
            visibility = ["//a:b"],
        )

        genrule(
            name = "hello_world",
            srcs = ["ignore_me.txt"],
            outs = ["message.txt"],
            cmd = 'echo "Hello, world." >message.txt',
        )
        """);
    Package pkg = getPackage("x");

    var testDep = pkg.getRule("dep");
    assertThat(testDep).hasSamePropertiesAs(roundTrip(testDep));

    var testRule = pkg.getRule("y");
    assertThat(testRule).hasSamePropertiesAs(roundTrip(testRule));

    var ccBinaryRule = pkg.getRule("cu");
    assertThat(ccBinaryRule).hasSamePropertiesAs(roundTrip(ccBinaryRule));

    // Covers the case of a native rule.
    var genruleRule = pkg.getRule("hello_world");
    assertThat(genruleRule).hasSamePropertiesAs(roundTrip(genruleRule));
  }

  private TargetData roundTrip(Target target) throws SerializationException, IOException {
    return RoundTripping.roundTrip(
        target.reduceForSerialization(),
        ImmutableClassToInstanceMap.of(
            RuleClassProvider.class, skyframeExecutor.getRuleClassProviderForTesting()));
  }
}
