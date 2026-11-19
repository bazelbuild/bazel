// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.query2.common.QueryPreloadingTestCase;
import com.google.devtools.build.lib.skyframe.TransitiveTargetValue.UnsuccessfulTransitiveTargetValue;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.SkyKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link TransitiveTargetFunction}.
 *
 * <p>Also see {@link QueryPreloadingTestCase} and descendants, which includes tests covering {@link
 * TransitiveTargetFunction}.
 */
@RunWith(JUnit4.class)
public final class TransitiveTargetFunctionTest extends BuildViewTestCase {

  private TransitiveTargetValue getSkyframeValueKeepGoing(String labelName) throws Exception {
    SkyKey skyKey = TransitiveTargetKey.of(Label.parseCanonical(labelName));
    return (TransitiveTargetValue)
        SkyframeExecutorTestUtils.evaluate(getSkyframeExecutor(), skyKey, true, reporter)
            .get(skyKey);
  }

  @Test
  public void testSyntaxErrorInDepPackage() throws Exception {
    reporter.removeHandler(failFastHandler);

    scratch.file(
        "a/BUILD",
        "genrule(name='x',",
        "        srcs = ['file.txt'],",
        "        outs = ['foo'],",
        "        cmd = 'echo')",
        "@"); // syntax error

    scratch.file(
        "b/BUILD",
        """
        genrule(
            name = "cc",
            outs = ["bar"],
            cmd = "echo",
            tools = ["//a:x"],
        )
        """);

    TransitiveTargetValue value = getSkyframeValueKeepGoing("//b:cc");

    // We should get error message from package a, but package b is properly loaded.
    assertContainsEvent("invalid character: '@'");
    assertThat(value.getTransitiveTargets().toList()).contains(Label.parseCanonical("//b:cc"));
    assertThat(value.encounteredLoadingError()).isTrue();
    assertThat(value).isInstanceOf(UnsuccessfulTransitiveTargetValue.class);
  }
}
