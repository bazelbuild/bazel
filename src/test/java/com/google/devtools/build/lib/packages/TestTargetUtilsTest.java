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

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.pkgcache.TargetProvider;
import com.google.devtools.build.lib.skyframe.TestSuiteExpansionValue;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Collection;
import java.util.EnumSet;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class TestTargetUtilsTest extends PackageLoadingTestCase {
  private Target test1;
  private Target test2;
  private Target test1b;
  private Target suite;

  @Before
  public final void createTargets() throws Exception {
    scratch.file(
        "tests/BUILD",
        "py_test(name = 'small_test_1',",
        "        srcs = ['small_test_1.py'],",
        "        data = [':xUnit'],",
        "        size = 'small',",
        "        tags = ['tag1'])",
        "",
        "sh_test(name = 'small_test_2',",
        "        srcs = ['small_test_2.sh'],",
        "        data = ['//testing/shbase:googletest.sh'],",
        "        size = 'small',",
        "        tags = ['tag2'])",
        "",
        "sh_test(name = 'large_test_1',",
        "        srcs = ['large_test_1.sh'],",
        "        data = ['//testing/shbase:googletest.sh', ':xUnit'],",
        "        size = 'large',",
        "        tags = ['tag1'])",
        "",
        "py_binary(name = 'notest',",
        "        srcs = ['notest.py'])",
        "cc_library(name = 'xUnit', data = ['//tools:test_sharding_compliant'])",
        "",
        "test_suite( name = 'smallTests', tags=['small'])");

    test1 = getTarget("//tests:small_test_1");
    test2 = getTarget("//tests:small_test_2");
    test1b = getTarget("//tests:large_test_1");
    suite = getTarget("//tests:smallTests");
  }

  @Test
  public void testFilterBySize() throws Exception {
    Predicate<Target> sizeFilter =
        TestTargetUtils.testSizeFilter(EnumSet.of(TestSize.SMALL, TestSize.LARGE));
    assertThat(sizeFilter.apply(test1)).isTrue();
    assertThat(sizeFilter.apply(test2)).isTrue();
    assertThat(sizeFilter.apply(test1b)).isTrue();
    sizeFilter = TestTargetUtils.testSizeFilter(EnumSet.of(TestSize.SMALL));
    assertThat(sizeFilter.apply(test1)).isTrue();
    assertThat(sizeFilter.apply(test2)).isTrue();
    assertThat(sizeFilter.apply(test1b)).isFalse();
  }

  @Test
  public void testFilterByTimeout() throws Exception {
    scratch.file(
        "timeouts/BUILD",
        "sh_test(name = 'long_timeout',",
        "          srcs = ['a.sh'],",
        "          size = 'small',",
        "          timeout = 'long')",
        "sh_test(name = 'short_timeout',",
        "          srcs = ['b.sh'],",
        "          size = 'small')",
        "sh_test(name = 'moderate_timeout',",
        "          srcs = ['c.sh'],",
        "          size = 'small',",
        "          timeout = 'moderate')");
    Target longTest = getTarget("//timeouts:long_timeout");
    Target shortTest = getTarget("//timeouts:short_timeout");
    Target moderateTest = getTarget("//timeouts:moderate_timeout");

    Predicate<Target> timeoutFilter =
        TestTargetUtils.testTimeoutFilter(EnumSet.of(TestTimeout.SHORT, TestTimeout.LONG));
    assertThat(timeoutFilter.apply(longTest)).isTrue();
    assertThat(timeoutFilter.apply(shortTest)).isTrue();
    assertThat(timeoutFilter.apply(moderateTest)).isFalse();
  }

  @Test
  public void testExpandTestSuites() throws Exception {
    assertExpandedSuites(Sets.newHashSet(test1, test2), Sets.newHashSet(test1, test2));
    assertExpandedSuites(Sets.newHashSet(test1, test2), Sets.newHashSet(suite));
    assertExpandedSuites(
        Sets.newHashSet(test1, test2, test1b), Sets.newHashSet(test1, suite, test1b));
    // The large test if returned as filtered from the test_suite rule, but should still be in the
    // result set as it's explicitly added.
    assertExpandedSuites(
        Sets.newHashSet(test1, test2, test1b), ImmutableSet.<Target>of(test1b, suite));
  }

  @Test
  public void testSkyframeExpandTestSuites() throws Exception {
    assertExpandedSuitesSkyframe(
        Sets.newHashSet(test1, test2), ImmutableSet.<Target>of(test1, test2));
    assertExpandedSuitesSkyframe(Sets.newHashSet(test1, test2), ImmutableSet.<Target>of(suite));
    assertExpandedSuitesSkyframe(
        Sets.newHashSet(test1, test2, test1b), ImmutableSet.<Target>of(test1, suite, test1b));
    // The large test if returned as filtered from the test_suite rule, but should still be in the
    // result set as it's explicitly added.
    assertExpandedSuitesSkyframe(
        Sets.newHashSet(test1, test2, test1b), ImmutableSet.<Target>of(test1b, suite));
  }

  @Test
  public void testExpandTestSuitesKeepGoing() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("broken/BUILD", "test_suite(name = 'broken', tests = ['//missing:missing_test'])");
    ResolvedTargets<Target> actual =
        TestTargetUtils.expandTestSuites(
            getPackageManager(),
            reporter,
            Sets.newHashSet(getTarget("//broken")),
            /*strict=*/ false,
            /* keepGoing= */ true);
    assertThat(actual.hasError()).isTrue();
    assertThat(actual.getTargets()).isEmpty();
  }

  private void assertExpandedSuites(Iterable<Target> expected, Collection<Target> suites)
      throws Exception {
    ResolvedTargets<Target> actual =
        TestTargetUtils.expandTestSuites(
            getPackageManager(), reporter, suites, /*strict=*/ false, /* keepGoing= */ true);
    assertThat(actual.hasError()).isFalse();
    assertThat(actual.getTargets()).containsExactlyElementsIn(expected);
  }

  private static final Function<Target, Label> TO_LABEL =
      new Function<Target, Label>() {
        @Override
        public Label apply(Target input) {
          return input.getLabel();
        }
      };

  private void assertExpandedSuitesSkyframe(Iterable<Target> expected, Collection<Target> suites)
      throws Exception {
    ImmutableSet<Label> suiteLabels = ImmutableSet.copyOf(Iterables.transform(suites, TO_LABEL));
    SkyKey key = TestSuiteExpansionValue.key(suiteLabels);
    EvaluationResult<TestSuiteExpansionValue> result =
        getSkyframeExecutor()
            .getDriverForTesting()
            .evaluate(ImmutableList.of(key), false, 1, reporter);
    ResolvedTargets<Target> actual = result.get(key).getTargets();
    assertThat(actual.hasError()).isFalse();
    assertThat(actual.getTargets()).containsExactlyElementsIn(expected);
  }

  @Test
  public void testExpandTestSuitesInterrupted() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("broken/BUILD", "test_suite(name = 'broken', tests = ['//missing:missing_test'])");
    try {
      TestTargetUtils.expandTestSuites(
          new TargetProvider() {
            @Override
            public Target getTarget(ExtendedEventHandler eventHandler, Label label)
                throws InterruptedException {
              throw new InterruptedException();
            }
          },
          reporter,
          Sets.newHashSet(getTarget("//broken")),
          /*strict=*/ false,
          /* keepGoing= */ true);
    } catch (TargetParsingException e) {
      assertThat(e).hasMessageThat().isNotNull();
    }
    assertThat(Thread.currentThread().isInterrupted()).isTrue();
  }
}
