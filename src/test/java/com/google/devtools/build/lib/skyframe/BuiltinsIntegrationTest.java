// Copyright 2020 The Bazel Authors. All rights reserved.
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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for builtins injection's integration with actual rule logic and our testing harness.
 *
 * <p>This suite is for test cases that want a normal rule class provider, unlike
 * BuiltinsInjectionTest, which has a minimal environment.
 */
@RunWith(JUnit4.class)
public class BuiltinsIntegrationTest extends BuildViewTestCase {

  // The _builtins_dummy symbol is used within BuiltinsIntegrationTest to confirm that
  // BuildViewTestCase sets up the builtins root properly, and by builtins_injection_test.sh to
  // confirm that --experimental_builtins_bzl_path is working. The dummy symbol should not be
  // exposed to user programs.
  @Test
  public void builtinsDummyIsNotAPublicApi() throws Exception {
    scratch.file("pkg/BUILD", "load(':foo.bzl', 'foo')");
    scratch.file("pkg/foo.bzl", "foo = _builtins_dummy");

    reporter.removeHandler(failFastHandler);
    Object result = getConfiguredTarget("//pkg:BUILD");
    assertThat(result).isNull();
    assertContainsEvent("_builtins_dummy is experimental");
  }

  @Test
  public void builtinsInjectionWorksInBuildViewTestCase() throws Exception {
    scratch.file("pkg/BUILD", "load(':foo.bzl', 'foo')");
    scratch.file("pkg/foo.bzl", "foo = 1; print(\"dummy :: \" + str(_builtins_dummy))");
    setBuildLanguageOptions(
        "--experimental_builtins_bzl_path=%bundled%", "--experimental_builtins_dummy=true");

    getConfiguredTarget("//pkg:BUILD");
    // The production builtins bzl code overwrites the dummy from "original value" to "overridden
    // value".
    assertContainsEvent("dummy :: overridden value");
  }
}
