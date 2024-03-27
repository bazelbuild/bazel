// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import java.util.Collection;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for ensuring that optimizations we have during package loading actually occur. */
@RunWith(JUnit4.class)
public class PackageLoadingOptimizationsTest extends PackageLoadingTestCase {
  @Test
  public void attributeListValuesAreDedupedIntraPackage() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        L = ["//other:t" + str(i) for i in range(10)]

        [sh_library(
            name = "t" + str(i),
            deps = L,
        ) for i in range(10)]
        """);

    Package fooPkg =
        getPackageManager()
            .getPackage(NullEventHandler.INSTANCE, PackageIdentifier.createInMainRepo("foo"));

    ImmutableList.Builder<ImmutableList<Label>> allListsBuilder = ImmutableList.builder();
    for (Rule ruleInstance : fooPkg.getTargets(Rule.class)) {
      assertThat(ruleInstance.getTargetKind()).isEqualTo("sh_library rule");
      @SuppressWarnings("unchecked")
      ImmutableList<Label> depsList = (ImmutableList<Label>) ruleInstance.getAttr("deps");
      allListsBuilder.add(depsList);
    }
    ImmutableList<ImmutableList<Label>> allLists = allListsBuilder.build();
    assertThat(allLists).hasSize(10);
    ImmutableList<Label> firstList = allLists.get(0);
    for (int i = 1; i < allLists.size(); i++) {
      assertThat(allLists.get(i)).isSameInstanceAs(firstList);
    }
  }

  @Test
  public void testRuntimeListValueIsDedupedAcrossRuleClasses() throws Exception {
    scratch.file(
        "foo/foo.bzl",
        """
        def _foo_test_impl(ctx):
            return

        foo_test = rule(implementation = _foo_test_impl, test = True)
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":foo.bzl", "foo_test")

        [sh_test(
            name = str(i) + "_test",
            srcs = ["t.sh"],
        ) for i in range(5)]

        [foo_test(name = str(i) + "_foo_test") for i in range(5)]
        """);

    Package fooPkg =
        getPackageManager()
            .getPackage(NullEventHandler.INSTANCE, PackageIdentifier.createInMainRepo("foo"));

    ImmutableList.Builder<ImmutableList<Label>> allListsBuilder = ImmutableList.builder();
    for (Rule ruleInstance : fooPkg.getTargets(Rule.class)) {
      assertThat(ruleInstance.getTargetKind()).endsWith("_test rule");
      @SuppressWarnings("unchecked")
      ImmutableList<Label> testRuntimeList =
          (ImmutableList<Label>) ruleInstance.getAttr("$test_runtime");
      allListsBuilder.add(testRuntimeList);
    }
    ImmutableList<ImmutableList<Label>> allLists = allListsBuilder.build();
    assertThat(allLists).hasSize(10);
    ImmutableList<Label> firstList = allLists.get(0);
    for (int i = 1; i < allLists.size(); i++) {
      assertThat(allLists.get(i)).isSameInstanceAs(firstList);
    }
  }

  @Test
  public void starlarkProviderIdentifierIsDedupedAcrossRuleClasses() throws Exception {
    scratch.file("foo/provider.bzl", "foo_provider = provider()");
    scratch.file(
        "foo/foo.bzl",
        """
        load(":provider.bzl", "foo_provider")

        def _foo_impl(ctx):
            return

        foo_rule = rule(implementation = _foo_impl, provides = [foo_provider])
        """);
    scratch.file(
        "foo/foobar.bzl",
        """
        load(":provider.bzl", "foo_provider")

        def _foobar_impl(ctx):
            return

        foobar_rule = rule(implementation = _foobar_impl, provides = [foo_provider])
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":foo.bzl", "foo_rule")
        load(":foobar.bzl", "foobar_rule")

        foo_rule(name = "foo_rule_instance")

        foobar_rule(name = "foobar_rule_instance")
        """);

    Package fooPkg =
        getPackageManager()
            .getPackage(NullEventHandler.INSTANCE, PackageIdentifier.createInMainRepo("foo"));

    ImmutableList.Builder<ImmutableList<StarlarkProviderIdentifier>> allListsBuilder =
        ImmutableList.builder();
    for (Rule ruleInstance : fooPkg.getTargets(Rule.class)) {
      RuleClass ruleClass = ruleInstance.getRuleClassObject();
      allListsBuilder.add(ruleClass.getAdvertisedProviders().getStarlarkProviders().asList());
    }
    ImmutableList<ImmutableList<StarlarkProviderIdentifier>> allLists = allListsBuilder.build();
    assertThat(allLists).hasSize(2);
    ImmutableList<StarlarkProviderIdentifier> firstList = allLists.get(0);
    for (int i = 1; i < allLists.size(); i++) {
      assertThat(allLists.get(i).get(0)).isSameInstanceAs(firstList.get(0));
    }
  }

  @Test
  public void testSuiteImplicitTestsAttributeValueIsSortedByTargetName() throws Exception {
    // When we have a BUILD file that instantiates some test targets
    scratch.file(
        "foo/BUILD",
        """
        # (in an order that is not target-name-order),
        sh_test(
            name = "bTest",
            srcs = ["test.sh"],
        )

        sh_test(
            name = "cTest",
            srcs = ["test.sh"],
        )

        sh_test(
            name = "aTest",
            srcs = ["test.sh"],
        )

        # And also a `test_suite` target, without setting the `test_suite.tests` attribute,
        test_suite(name = "suite")
        """);

    // Then when we load the package,
    PackageIdentifier fooPkgId = PackageIdentifier.createInMainRepo("foo");
    Package fooPkg = getPackageManager().getPackage(NullEventHandler.INSTANCE, fooPkgId);

    // And we get the Rule instance for the `test_suite` target,
    Rule testSuiteRuleInstance = (Rule) fooPkg.getTarget("suite");
    assertThat(testSuiteRuleInstance.getTargetKind()).isEqualTo("test_suite rule");
    @SuppressWarnings("unchecked")
    Collection<Label> implicitTestsAttributeValue =
        (Collection<Label>) testSuiteRuleInstance.getAttr("$implicit_tests");
    // The $implicit_tests attribute's value is ordered by target-name.
    assertThat(implicitTestsAttributeValue)
        .containsExactly(
            Label.create(fooPkgId, "aTest"),
            Label.create(fooPkgId, "bTest"),
            Label.create(fooPkgId, "cTest"))
        .inOrder();
  }
}
