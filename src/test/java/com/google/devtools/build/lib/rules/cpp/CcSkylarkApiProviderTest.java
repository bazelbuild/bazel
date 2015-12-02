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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCaseForJunit4;
import com.google.devtools.build.lib.testutil.TestConstants;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for Skylark providers for cpp rules.
 */
@RunWith(JUnit4.class)
public class CcSkylarkApiProviderTest extends BuildViewTestCaseForJunit4 {
  private CcSkylarkApiProvider getApi(String label) throws Exception {
    RuleConfiguredTarget rule = (RuleConfiguredTarget) getConfiguredTarget(label);
    return (CcSkylarkApiProvider) rule.get(CcSkylarkApiProvider.NAME);
  }

  @Test
  public void testTransitiveHeaders() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "cc_binary(",
        "    name = 'check',",
        "    srcs = ['bin.cc', 'bin.h'],",
        "    deps = [':check_lib'],",
        ")",
        "cc_library(",
        "    name = 'check_lib',",
        "    srcs = ['lib.cc', 'lib.h'],",
        ")");
    assertThat(ActionsTestUtil.baseArtifactNames(getApi("//pkg:check").getTransitiveHeaders()))
        .containsAllOf("lib.h", "bin.h");
    assertThat(ActionsTestUtil.baseArtifactNames(getApi("//pkg:check_lib").getTransitiveHeaders()))
        .contains("lib.h");
  }

  @Test
  public void testLinkFlags() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "cc_binary(",
        "    name = 'check',",
        "    srcs = ['bin.cc', 'bin.h'],",
        "    linkopts = ['-lm'],",
        "    deps = [':dependent_lib'],",
        ")",
        "cc_binary(",
        "    name = 'check_no_srcs',",
        "    linkopts = ['-lm'],",
        "    deps = [':dependent_lib'],",
        ")",
        "cc_library(",
        "    name = 'dependent_lib',",
        "    linkopts = ['-lz'],",
        "    deps = [':check_lib'],",
        ")",
        "cc_library(",
        "    name = 'check_lib',",
        "    defines = ['foo'],",
        "    linkopts = ['-Wl,-M'],",
        ")");
    assertThat(getApi("//pkg:check_lib").getLinkopts())
        .contains("-Wl,-M");
    assertThat(getApi("//pkg:dependent_lib").getLinkopts())
        .containsAllOf("-lz", "-Wl,-M")
        .inOrder();
    assertThat(getApi("//pkg:check").getLinkopts())
        .isEmpty();
    assertThat(getApi("//pkg:check_no_srcs").getLinkopts())
        .isEmpty();
  }

  @Test
  public void testLibraries() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "cc_binary(",
        "    name = 'check',",
        "    srcs = ['bin.cc', 'bin.h'],",
        "    deps = [':check_lib'],",
        ")",
        "cc_binary(",
        "    name = 'check_no_srcs',",
        "    deps = [':check_lib'],",
        ")",
        "cc_library(",
        "    name = 'check_lib',",
        "    srcs = ['lib.cc', 'lib.h'],",
        ")");
    assertThat(ActionsTestUtil.baseArtifactNames(getApi("//pkg:check_lib").getLibraries()))
        .containsExactly("libcheck_lib.a");
    assertThat(ActionsTestUtil.baseArtifactNames(getApi("//pkg:check").getLibraries()))
        .isEmpty();
    assertThat(ActionsTestUtil.baseArtifactNames(getApi("//pkg:check_no_srcs").getLibraries()))
        .isEmpty();
  }

  @Test
  public void testCcFlags() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "cc_binary(",
        "    name = 'check',",
        "    srcs = ['bin.cc', 'bin.h'],",
        "    deps = [':check_lib'],",
        ")",
        "cc_library(",
        "    name = 'check_lib',",
        "    defines = ['foo'],",
        ")");
    // The particular values for include directories are slightly
    // fragile because the build system changes. But check for at
    // least one normal include, one system include, and one define.
    assertThat(getApi("//pkg:check").getCcFlags())
        .containsAllOf("-iquote .", "-isystem " + TestConstants.GCC_INCLUDE_PATH, "-Dfoo");
  }
}
