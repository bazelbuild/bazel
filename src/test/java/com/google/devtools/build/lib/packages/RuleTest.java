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
import static com.google.devtools.build.lib.packages.util.TargetDataSubject.assertThat;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Rule}. */
@RunWith(JUnit4.class)
public class RuleTest extends PackageLoadingTestCase {

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
        cc_test(
            name = "y",
            srcs = ["a"],
            local = 0,
        )

        cc_test(
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
        cc_test(name = "y")

        cc_test(
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
        cc_binary(
            name = "pr",
            visibility = ["//visibility:private"],
        )

        cc_binary(
            name = "pu",
            visibility = ["//visibility:public"],
        )

        cc_binary(
            name = "cu",
            visibility = ["//a:b"],
        )
        """);
    Package pkg = getTarget("//x:BUILD").getPackage();
    assertThat(pkg.getRule("pu").getVisibility()).isEqualTo(RuleVisibility.PUBLIC);
    assertThat(pkg.getRule("pr").getVisibility()).isEqualTo(RuleVisibility.PRIVATE);
  }

  @Test
  public void testReduceForSerialization() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        cc_library(
            name = "dep",
            deprecation = "message should serialize",
        )

        cc_test(
            name = "y",
            srcs = ["a"],
            deps = [":dep"],
        )

        cc_binary(
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
    Package pkg = getTarget("//x:BUILD").getPackage();

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
    return TestUtils.roundTrip(
        target.reduceForSerialization(),
        ImmutableClassToInstanceMap.of(
            RuleClassProvider.class, skyframeExecutor.getRuleClassProviderForTesting()));
  }
}
