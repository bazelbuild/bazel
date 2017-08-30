// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for propagation of the presence of categories of source in the transitive deps of a target.
 */
@RunWith(JUnit4.class)
public class TransitiveSourcesTest extends ObjcRuleTestCase {

  @Before
  public void setup() throws Exception {
    MockObjcSupport.setup(
        mockToolsConfig,
        "feature {",
        "  name: 'contains_objc_source'",
        "  flag_set {",
        "    flag_group {",
        "      flag: 'DUMMY_FLAG'",
        "    }",
        "    action: 'c++-compile'",
        "    action: 'c++-link-executable'",
        "    action: 'objc++-executable'",
        "  }",
        "}");
    useConfiguration();
  }

  private CppCompileAction getCppCompileAction(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    return actionsTestUtil()
        .findTransitivePrerequisitesOf(
            getFilesToBuild(target).iterator().next(), CppCompileAction.class)
        .get(0);
  }

  private CppLinkAction getExecutableCppLinkAction(String label) throws Exception {
    return (CppLinkAction) getGeneratingAction(getExecutable(label));
  }

  @Test
  public void testCcBinaryTransitiveObjcFeature() throws Exception {
    scratch.file("bottom/BUILD", "objc_library(name='lib', srcs=['a.m'])");
    scratch.file("middle/BUILD", "cc_library(name='lib', srcs=['a.cc'], deps=['//bottom:lib'])");
    scratch.file("top/BUILD", "cc_binary(name='bin', srcs=['a.cc'], deps=['//middle:lib'])");

    assertThat(getCppCompileAction("//top:bin").getArguments()).contains("DUMMY_FLAG");
    assertThat(getExecutableCppLinkAction("//top:bin").getArguments()).contains("DUMMY_FLAG");
  }

  @Test
  public void testAppleBinaryTransitiveObjcFeature() throws Exception {
    scratch.file("bottom/BUILD", "objc_library(name='lib', srcs=['a.m'])");
    scratch.file("middle/BUILD", "cc_library(name='lib', srcs=['a.cc'], deps=['//bottom:lib'])");
    scratch.file(
        "top/BUILD",
        "apple_binary(name='bin', platform_type='ios', srcs=['a.cc'], deps=['//middle:lib'])");

    assertThat(getCppCompileAction("//top:bin").getArguments()).contains("DUMMY_FLAG");
    assertThat(linkAction("//top:bin").getArguments()).contains("DUMMY_FLAG");
  }
}
