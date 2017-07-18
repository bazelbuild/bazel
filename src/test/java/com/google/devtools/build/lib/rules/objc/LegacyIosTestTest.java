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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Legacy test: These tests test --experimental_objc_crosstool=off. See README.
 */
@RunWith(JUnit4.class)
public class LegacyIosTestTest extends IosTestTest {
  @Override
  protected ObjcCrosstoolMode getObjcCrosstoolMode() {
    return ObjcCrosstoolMode.OFF;
  }

  @Test
  public void testGetsIncludesFromTestRig() throws Exception {
    scratch.file("x/BUILD",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        "    includes = ['libinc'],",
        "    sdk_includes = ['libinc_sdk'],",
        ")",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    includes = ['bininc'],",
        "    sdk_includes = ['bininc_sdk'],",
        "    deps = [':lib'],",
        ")",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = ':bin',",
        ")",
        "ios_test(",
        "    name = 'test',",
        "    srcs = ['test.m'],",
        "    includes = ['testinc'],",
        "    sdk_includes = ['testinc_sdk'],",
        "    xctest = 1,",
        "    xctest_app = ':testApp',",
        ")");
    List<String> compileArgs = compileAction("//x:test", "test.o").getArguments();
    assertContainsSublist(compileArgs, ImmutableList.of("-I", "x/libinc"));
    assertContainsSublist(compileArgs, ImmutableList.of("-I", "x/bininc"));
    assertContainsSublist(compileArgs, ImmutableList.of("-I", "x/testinc"));

    String sdkIncludeDir = AppleToolchain.sdkDir() + "/usr/include/";
    assertContainsSublist(compileArgs, ImmutableList.of("-I", sdkIncludeDir + "libinc_sdk"));
    assertContainsSublist(compileArgs, ImmutableList.of("-I", sdkIncludeDir + "bininc_sdk"));
    assertContainsSublist(compileArgs, ImmutableList.of("-I", sdkIncludeDir + "testinc_sdk"));
  }

  @Test
  public void testGetsFrameworksFromTestRig() throws Exception {
    scratch.file("x/BUILD",
        "objc_framework(",
        "    name = 'fx',",
        "    framework_imports = ['fx.framework/1'],",
        ")",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['lib.m'],",
        "    deps = [':fx'],",
        ")",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    deps = [':lib'],",
        ")",
        "ios_application(",
        "    name = 'testApp',",
        "    binary = ':bin',",
        ")",
        "ios_test(",
        "    name = 'test',",
        "    srcs = ['test.m'],",
        "    xctest = 1,",
        "    xctest_app = ':testApp',",
        ")");
    CommandAction compileAction = compileAction("//x:test", "test.o");

    assertThat(Artifact.toExecPaths(compileAction.getInputs()))
        .contains("x/fx.framework/1");
    assertContainsSublist(compileAction.getArguments(), ImmutableList.of("-F", "x"));

    CommandAction linkAction = linkAction("//x:test");
    assertThat(Joiner.on(" ").join(linkAction.getArguments())).doesNotContain("-framework fx");
  }

  @Test
  public void testReceivesTransitivelyPropagatedDefines() throws Exception {
    checkReceivesTransitivelyPropagatedDefines(RULE_TYPE);
  }

  @Test
  public void testSdkIncludesUsedInCompileAction() throws Exception {
    checkSdkIncludesUsedInCompileAction(RULE_TYPE);
  }
}
