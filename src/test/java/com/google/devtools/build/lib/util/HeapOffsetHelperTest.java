// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase.RecordingBugReporter;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.MemoryPressureModule;
import com.google.devtools.build.lib.runtime.MemoryPressureOptions;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests HeapOffsetHelper and verify we are properly pulling heap data. */
@RunWith(JUnit4.class)
public final class HeapOffsetHelperTest extends BuildIntegrationTestCase {
  private final MemoryPressureModule memoryPressureModule = new MemoryPressureModule();

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder().addBlazeModule(memoryPressureModule);
  }

  @Before
  public void writeTrivialFooTarget() throws Exception {
    write(
        "foo/BUILD",
        """
        genrule(
            name = "foo",
            outs = ["out"],
            cmd = "touch $@",
        )
        """);
  }

  @Test
  public void testBadPattern() throws Exception {
    RecordingBugReporter bugReporter = recordBugReportsAndReinitialize();

    // short-circuit this test when our version isn't JDK 21 or in Bazel
    // environments where JDK 21 doesn't have this.
    if (!HeapOffsetHelper.isWorkaroundNeeded() || AnalysisMock.get().isThisBazel()) {
      return;
    }

    buildTarget("//foo:foo");

    Pattern badPattern = Pattern.compile("horse");

    long offset = HeapOffsetHelper.getSizeOfFillerArrayOnHeap(badPattern, bugReporter);
    assertThat(offset).isEqualTo(0);

    assertThat(bugReporter.getExceptions()).hasSize(1);
    Throwable reported = Iterables.getOnlyElement(bugReporter.getExceptions());

    assertThat(reported).isInstanceOf(IllegalStateException.class);
    assertThat(reported).hasMessageThat().contains("JDK 21");
  }

  @Test
  public void matchesOpenJdk21Filler() throws Exception {
    // short-circuit this test when our version isn't JDK 21 or in Bazel
    // environments where JDK 21 doesn't have this.
    if (!HeapOffsetHelper.isWorkaroundNeeded() || AnalysisMock.get().isThisBazel()) {
      return;
    }

    // NOTE: If this test fails, it means that the JDK has changed and we need to update the
    // pattern in MemoryPressureOptions.  The flag can also be set in rc files to override the
    // default before a release.
    RecordingBugReporter bugReporter = recordBugReportsAndReinitialize();
    buildTarget("//foo:foo");

    Pattern defaultOptionPattern =
        getRuntimeWrapper()
            .getCommandEnvironment()
            .getOptions()
            .getOptions(MemoryPressureOptions.class)
            .jvmHeapHistogramInternalObjectPattern
            .regexPattern();

    long offset = HeapOffsetHelper.getSizeOfFillerArrayOnHeap(defaultOptionPattern, bugReporter);
    bugReporter.assertNoExceptions();

    assertThat(offset).isGreaterThan(0);
  }
}
