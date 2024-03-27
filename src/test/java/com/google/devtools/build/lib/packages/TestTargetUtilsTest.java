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
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.TestFilter;
import com.google.devtools.build.lib.skyframe.TestsForTargetPatternValue;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Collection;
import java.util.EnumSet;
import java.util.function.Predicate;
import net.starlark.java.syntax.Location;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class TestTargetUtilsTest extends PackageLoadingTestCase {
  private Target test1;
  private Target test2;
  private Target test1b;
  private Target suite;

  @Before
  public void createTargets() throws Exception {
    scratch.file(
        "tests/BUILD",
        """
        sh_test(
            name = "small_test_1",
            size = "small",
            srcs = ["small_test_1.sh"],
            data = [":xUnit"],
            tags = ["tag1"],
        )

        sh_test(
            name = "small_test_2",
            size = "small",
            srcs = ["small_test_2.sh"],
            data = ["//testing/shbase:googletest.sh"],
            tags = ["tag2"],
        )

        sh_test(
            name = "large_test_1",
            size = "large",
            srcs = ["large_test_1.sh"],
            data = [
                ":xUnit",
                "//testing/shbase:googletest.sh",
            ],
            tags = ["tag1"],
        )

        sh_binary(
            name = "notest",
            srcs = ["notest.sh"],
        )

        cc_library(name = "xUnit")

        test_suite(
            name = "smallTests",
            tags = ["small"],
        )
        """);

    test1 = getTarget("//tests:small_test_1");
    test2 = getTarget("//tests:small_test_2");
    test1b = getTarget("//tests:large_test_1");
    suite = getTarget("//tests:smallTests");
  }

  @Test
  public void testFilterBySize() {
    Predicate<Target> sizeFilter =
        TestFilter.testSizeFilter(EnumSet.of(TestSize.SMALL, TestSize.LARGE));
    assertThat(sizeFilter.test(test1)).isTrue();
    assertThat(sizeFilter.test(test2)).isTrue();
    assertThat(sizeFilter.test(test1b)).isTrue();
    sizeFilter = TestFilter.testSizeFilter(EnumSet.of(TestSize.SMALL));
    assertThat(sizeFilter.test(test1)).isTrue();
    assertThat(sizeFilter.test(test2)).isTrue();
    assertThat(sizeFilter.test(test1b)).isFalse();
  }

  @Test
  public void testFilterByLang() {
    LoadingOptions options = new LoadingOptions();
    options.testLangFilterList = ImmutableList.of("positive", "-negative");
    options.testSizeFilterSet = ImmutableSet.of();
    options.testTimeoutFilterSet = ImmutableSet.of();
    options.testTagFilterList = ImmutableList.of();
    TestFilter filter = TestFilter.forOptions(options);
    Package pkg = mock(Package.class);
    RuleClass ruleClass = mock(RuleClass.class);
    when(ruleClass.getDefaultImplicitOutputsFunction())
        .thenReturn(SafeImplicitOutputsFunction.NONE);
    Rule mockRule =
        new Rule(
            pkg,
            Label.parseCanonicalUnchecked("//pkg:a"),
            ruleClass,
            Location.fromFile(""),
            /* interiorCallStack= */ null);
    when(ruleClass.getName()).thenReturn("positive_test");
    assertThat(filter.apply(mockRule)).isTrue();
    when(ruleClass.getName()).thenReturn("negative_test");
    assertThat(filter.apply(mockRule)).isFalse();
  }

  @Test
  public void testFilterByTimeout() throws Exception {
    scratch.file(
        "timeouts/BUILD",
        """
        sh_test(
            name = "long_timeout",
            size = "small",
            timeout = "long",
            srcs = ["a.sh"],
        )

        sh_test(
            name = "short_timeout",
            size = "small",
            srcs = ["b.sh"],
        )

        sh_test(
            name = "moderate_timeout",
            size = "small",
            timeout = "moderate",
            srcs = ["c.sh"],
        )
        """);
    Target longTest = getTarget("//timeouts:long_timeout");
    Target shortTest = getTarget("//timeouts:short_timeout");
    Target moderateTest = getTarget("//timeouts:moderate_timeout");

    Predicate<Target> timeoutFilter =
        TestFilter.testTimeoutFilter(EnumSet.of(TestTimeout.SHORT, TestTimeout.LONG));
    assertThat(timeoutFilter.test(longTest)).isTrue();
    assertThat(timeoutFilter.test(shortTest)).isTrue();
    assertThat(timeoutFilter.test(moderateTest)).isFalse();
  }

  @Test
  public void testSkyframeExpandTestSuites() throws Exception {
    assertExpandedSuitesSkyframe(Sets.newHashSet(test1, test2), ImmutableSet.of(test1, test2));
    assertExpandedSuitesSkyframe(Sets.newHashSet(test1, test2), ImmutableSet.of(suite));
    assertExpandedSuitesSkyframe(
        Sets.newHashSet(test1, test2, test1b), ImmutableSet.of(test1, suite, test1b));
    // The large test if returned as filtered from the test_suite rule, but should still be in the
    // result set as it's explicitly added.
    assertExpandedSuitesSkyframe(
        Sets.newHashSet(test1, test2, test1b), ImmutableSet.of(test1b, suite));
  }

  @Test
  public void testSortTagsBySenseSeparatesTagsNaively() {
    // Contrived, but intentional.
    Pair<Collection<String>, Collection<String>> result =
        TestTargetUtils.sortTagsBySense(
            ImmutableList.of("tag1", "tag2", "tag3", "-tag1", "+tag2", "-tag3"));

    assertThat(result.first).containsExactly("tag1", "tag2", "tag3");
    assertThat(result.second).containsExactly("tag1", "tag3");
  }

  private void assertExpandedSuitesSkyframe(Iterable<Target> expected, Collection<Target> suites)
      throws Exception {
    ImmutableSet<Label> expectedLabels =
        ImmutableSet.copyOf(Iterables.transform(expected, Target::getLabel));
    ImmutableSet<Label> suiteLabels =
        ImmutableSet.copyOf(Iterables.transform(suites, Target::getLabel));
    SkyKey key = TestsForTargetPatternValue.key(suiteLabels);
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setParallelism(1)
            .setEventHandler(reporter)
            .build();
    EvaluationResult<TestsForTargetPatternValue> result =
        getSkyframeExecutor().getEvaluator().evaluate(ImmutableList.of(key), evaluationContext);
    ResolvedTargets<Label> actual = result.get(key).getLabels();
    assertThat(actual.hasError()).isFalse();
    assertThat(actual.getTargets()).containsExactlyElementsIn(expectedLabels);
  }
}
