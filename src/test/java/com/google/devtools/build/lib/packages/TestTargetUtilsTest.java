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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.TestFilter;
import com.google.devtools.build.lib.skyframe.TestsForTargetPatternValue;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Collection;
import java.util.EnumSet;
import java.util.function.Predicate;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

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
  public void testFilterByLang() throws Exception {
    StoredEventHandler eventHandler = new StoredEventHandler();
    LoadingOptions options = new LoadingOptions();
    options.testLangFilterList = ImmutableList.of("nonexistent", "existent", "-noexist", "-exist");
    options.testSizeFilterSet = ImmutableSet.of();
    options.testTimeoutFilterSet = ImmutableSet.of();
    options.testTagFilterList = ImmutableList.of();
    TestFilter filter =
        TestFilter.forOptions(
            options, eventHandler, ImmutableSet.of("existent_test", "exist_test"));
    assertThat(eventHandler.getEvents()).hasSize(2);
    Package pkg = Mockito.mock(Package.class);
    RuleClass ruleClass = Mockito.mock(RuleClass.class);
    Rule mockRule =
        new Rule(
            pkg,
            null,
            ruleClass,
            Location.fromFile(""),
            CallStack.EMPTY,
            new AttributeContainer(ruleClass));
    Mockito.when(ruleClass.getName()).thenReturn("existent_library");
    assertThat(filter.apply(mockRule)).isTrue();
    Mockito.when(ruleClass.getName()).thenReturn("exist_library");
    assertThat(filter.apply(mockRule)).isFalse();
    assertThat(eventHandler.getEvents())
        .contains(Event.warn("Unknown language 'nonexistent' in --test_lang_filters option"));
    assertThat(eventHandler.getEvents())
        .contains(Event.warn("Unknown language 'noexist' in --test_lang_filters option"));
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
        TestFilter.testTimeoutFilter(EnumSet.of(TestTimeout.SHORT, TestTimeout.LONG));
    assertThat(timeoutFilter.test(longTest)).isTrue();
    assertThat(timeoutFilter.test(shortTest)).isTrue();
    assertThat(timeoutFilter.test(moderateTest)).isFalse();
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

  private static final Function<Target, Label> TO_LABEL =
      new Function<Target, Label>() {
        @Override
        public Label apply(Target input) {
          return input.getLabel();
        }
      };

  private void assertExpandedSuitesSkyframe(Iterable<Target> expected, Collection<Target> suites)
      throws Exception {
    ImmutableSet<Label> expectedLabels =
        ImmutableSet.copyOf(Iterables.transform(expected, TO_LABEL));
    ImmutableSet<Label> suiteLabels = ImmutableSet.copyOf(Iterables.transform(suites, TO_LABEL));
    SkyKey key = TestsForTargetPatternValue.key(suiteLabels);
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(1)
            .setEventHandler(reporter)
            .build();
    EvaluationResult<TestsForTargetPatternValue> result =
        getSkyframeExecutor().getDriver().evaluate(ImmutableList.of(key), evaluationContext);
    ResolvedTargets<Label> actual = result.get(key).getLabels();
    assertThat(actual.hasError()).isFalse();
    assertThat(actual.getTargets()).containsExactlyElementsIn(expectedLabels);
  }
}
