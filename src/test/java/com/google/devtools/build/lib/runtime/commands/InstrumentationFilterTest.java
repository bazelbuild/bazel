// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.buildtool.InstrumentationFilterSupport;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.packages.Target;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for --instrumentation_filter heuristic in the {@link CoverageCommand} class. */
@RunWith(JUnit4.class)
public class InstrumentationFilterTest extends BuildViewTestCase {

  private EventCollector events;

  @Before
  public void removeFastFailHandler() {
    reporter.removeHandler(failFastHandler);
  }

  @Before
  public void initializeEvents() {
    events = new EventCollector(EventKind.INFO);
  }

  private List<Target> getTargets(String... labels) throws Exception {
    List<Target> targets = new ArrayList<>();
    for (String label : labels) {
      targets.add(getTarget(label));
    }
    return targets;
  }

  private void assertEventsReportInstrumentationFilter(String expectedFilter) {
    List<String> messages = new ArrayList<>();
    for (Event event : events) {
      messages.add(event.getMessage());
    }
    assertThat(messages)
        .containsExactly(
            String.format(
                "Using default value for --instrumentation_filter: \"%s\".", expectedFilter),
            "Override the above default with --instrumentation_filter");
  }

  @Test
  public void testSingleTest() throws Exception {
    scratch.file(
        "my/package1/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t1',srcs=['t1.sh'])");
    List<Target> targets = getTargets("//my/package1:t1");
    String expectedFilter = "^//my/package1[/:]";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
    assertEventsReportInstrumentationFilter(expectedFilter);
  }

  @Test
  public void testAllTestsInPackage() throws Exception {
    scratch.file(
        "foo/test/BUILD",
        """
        load('//test_defs:foo_test.bzl', 'foo_test')
        foo_test(
            name = "t1",
            srcs = ["t1.sh"],
        )

        foo_test(
            name = "t2",
            srcs = ["t1.sh"],
        )
        """);
    List<Target> targets = getTargets("//foo/test:t1", "//foo/test:t2");
    String expectedFilter = "^//foo/test[/:]";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
    assertEventsReportInstrumentationFilter(expectedFilter);
  }

  @Test
  public void testMultiplePackages() throws Exception {
    scratch.file(
        "my/package1/BUILD",
        """
        load('//test_defs:foo_test.bzl', 'foo_test')
        foo_test(
            name = "t1",
            srcs = ["t1.sh"],
        )

        test_suite(
            name = "ts",
            tests = [
                "//other/package1:t1",
                "//other/package2:ts",
            ],
        )
        """);
    scratch.file(
        "other/package1/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t1',srcs=['t1.sh'])");
    scratch.file("other/package2/BUILD", "test_suite(name='ts', tests=['//other/package3:t3'])");
    scratch.file(
        "other/package3/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t3',srcs=['t3.sh'])");
    List<Target> targets = getTargets("//my/package1:t1", "//other/package1:t1");
    String expectedFilter = "^//my/package1[/:],^//other/package1[/:]";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
    assertEventsReportInstrumentationFilter(expectedFilter);
  }

  @Test
  public void testTestSuiteExpansion() throws Exception {
    scratch.file(
        "my/package1/BUILD",
        "test_suite(name='ts', tests=['//other/package1:t1', '//other/package2:ts'])");
    scratch.file(
        "other/package1/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t1',srcs=['t1.sh'])");
    scratch.file("other/package2/BUILD", "test_suite(name='ts', tests=['//other/package3:t3'])");
    scratch.file(
        "other/package3/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t3',srcs=['t3.sh'])");
    List<Target> targets = getTargets("//my/package1:ts");
    String expectedFilter = "^//my/package1[/:],^//other/package1[/:],^//other/package2[/:]";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
    assertEventsReportInstrumentationFilter(expectedFilter);
  }

  @Test
  public void testParentAndChildPackageCombined() throws Exception {
    scratch.file(
        "parent/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t1', srcs=['t1.sh'])");
    scratch.file(
        "parent/child/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t2', srcs=['t2.sh'])");
    List<Target> targets = getTargets("//parent:t1", "//parent/child:t2");
    String expectedFilter = "^//parent[/:]";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
    assertEventsReportInstrumentationFilter(expectedFilter);
  }

  @Test
  public void testJavascriptTests() throws Exception {
    scratch.file(
        "javascript/other/tests/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t1',srcs=['t1.sh'])");
    List<Target> targets = getTargets("//javascript/other/tests:t1");
    String expectedFilter = "^//javascript/other[/:]";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
    assertEventsReportInstrumentationFilter(expectedFilter);
  }

  @Test
  public void testJavaTests() throws Exception {
    scratch.file(
        "javatests/other/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t1',srcs=['t1.sh'])");
    List<Target> targets = getTargets("//javatests/other:t1");
    String expectedFilter = "^//java/other[/:]";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
    assertEventsReportInstrumentationFilter(expectedFilter);
  }

  @Test
  public void testInternal() throws Exception {
    scratch.file(
        "another/internal/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t1',srcs=['t1.sh'])");
    List<Target> targets = getTargets("//another/internal:t1");
    String expectedFilter = "^//another[/:]";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
    assertEventsReportInstrumentationFilter(expectedFilter);
  }

  @Test
  public void testPublic() throws Exception {
    scratch.file(
        "another/public/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t1',srcs=['t1.sh'])");
    List<Target> targets = getTargets("//another/public:t1");
    String expectedFilter = "^//another[/:]";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
    assertEventsReportInstrumentationFilter(expectedFilter);
  }

  @Test
  public void testIncludesTopLevel() throws Exception {
    scratch.file(
        "BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t1',srcs=['t1.sh'])");
    scratch.file(
        "foo/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t1',srcs=['t1.sh'])");
    InstrumentationFilterSupport.computeInstrumentationFilter(
        events, getTargets("//:t1", "//foo:t1"));
    assertEventsReportInstrumentationFilter("^//");
    List<Target> targets = getTargets();
    String expectedFilter = "^//";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
    assertEventsReportInstrumentationFilter(expectedFilter);
  }

  @Test
  public void testTopLevelOnly() throws Exception {
    scratch.file(
        "BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='t1',srcs=['t1.sh'])");
    List<Target> targets = getTargets("//:t1");
    String expectedFilter = "^//";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
    assertEventsReportInstrumentationFilter(expectedFilter);
  }

  @Test
  public void testNoTests() throws Exception {
    List<Target> targets = getTargets();
    String expectedFilter = "^//";
    assertThat(InstrumentationFilterSupport.computeInstrumentationFilter(events, targets))
        .isEqualTo(expectedFilter);
    // If there are no targets, this doesn't get output at all.
    assertThat(events).isEmpty();
  }
}
