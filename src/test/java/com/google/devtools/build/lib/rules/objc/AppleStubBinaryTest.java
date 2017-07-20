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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.testutil.Scratch;
import java.io.IOException;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for apple_stub_binary. */
@RunWith(JUnit4.class)
public class AppleStubBinaryTest extends ObjcRuleTestCase {

  static final RuleType RULE_TYPE =
      new RuleType("apple_stub_binary") {
        @Override
        Iterable<String> requiredAttributes(
            Scratch scratch, String packageDir, Set<String> alreadyAdded) throws IOException {
          ImmutableList.Builder<String> attributes = new ImmutableList.Builder<>();
          if (!alreadyAdded.contains("platform_type")) {
            attributes.add("platform_type = 'ios'");
          }
          return attributes.build();
        }
      };

  @Test
  public void testCopyActionEnv() throws Exception {
    RULE_TYPE.scratchTarget(
        scratch,
        "xcenv_based_path",
        "'$(SDKROOT)/Library/Application Support/WatchKit/WK'",
        "platform_type",
        "'watchos'");

    useConfiguration(
        "--watchos_cpus=i386,armv7k", "--xcode_version=7.3", "--watchos_sdk_version=2.1");

    CommandAction action = (CommandAction) lipoBinAction("//x:x");
    assertAppleSdkVersionEnv(action, "2.1");
    assertAppleSdkPlatformEnv(action, "WatchOS");
    assertXcodeVersionEnv(action, "7.3");
  }

  @Test
  public void testFailsWithUndefinedVar() throws Exception {
    String target =
        RULE_TYPE.target(
            scratch,
            "x",
            "x",
            "xcenv_based_path",
            "'$(NOT_ALLOWED)/Library/Application Support/WatchKit/WK'");
    useConfiguration("--xcode_version=7.3");

    checkError("x", "x", "The stub binary path must be rooted at", target);
  }

  @Test
  public void testFailsIfPathDoesNotBeginWithVar() throws Exception {
    String target =
        RULE_TYPE.target(
            scratch, "x", "x", "xcenv_based_path", "'/Library/Application Support/WatchKit/WK'");
    useConfiguration("--xcode_version=7.3");

    checkError("x", "x", "The stub binary path must be rooted at", target);
  }

  @Test
  public void testFailsWithUnnormalizedPath() throws Exception {
    String target =
        RULE_TYPE.target(
            scratch,
            "x",
            "x",
            "xcenv_based_path",
            "'$(SDKROOT)/../Library/Application Support/WatchKit/WK'");
    useConfiguration("--xcode_version=7.3");

    checkError("x", "x", AppleStubBinary.PATH_NOT_NORMALIZED_ERROR, target);
  }

  @Test
  public void testPlatformSelectionIos() throws Exception {
    checkObjcPropagatedResourcesRespectPlatform(PlatformType.IOS);
  }

  @Test
  public void testPlatformSelectionTvos() throws Exception {
    checkObjcPropagatedResourcesRespectPlatform(PlatformType.TVOS);
  }

  private void checkObjcPropagatedResourcesRespectPlatform(PlatformType platformType)
      throws Exception {
    String platformName = platformType.toString();

    scratch.file("x/a.m");
    scratch.file("x/ios.txt");
    scratch.file("x/tvos.txt");
    scratch.file(
        "x/BUILD",
        "apple_stub_binary(",
        "    name = 'bin',",
        "    platform_type = '" + platformName + "',",
        "    xcenv_based_path = '$(SDKROOT)/foo/bar',",
        "    deps = [':lib'],",
        ")",
        "",
        "config_setting(name = 'ios', values = {'apple_platform_type': 'ios'})",
        "config_setting(name = 'tvos', values = {'apple_platform_type': 'tvos'})",
        "",
        "objc_library(",
        "    name = 'lib',",
        "    srcs = ['a.m'],",
        "    resources = select({",
        "        ':ios': ['ios.txt'],",
        "        ':tvos': ['tvos.txt'],",
        "    })",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    ObjcProvider objc = (ObjcProvider) target.get(ObjcProvider.SKYLARK_NAME);

    // The propagated objc provider should only contain one file, and that file is the one selected
    // for the given platform type.
    NestedSet<BundleableFile> bundleFiles = objc.get(ObjcProvider.BUNDLE_FILE);
    assertThat(bundleFiles).hasSize(1);

    BundleableFile bundleFile = bundleFiles.iterator().next();
    assertThat(bundleFile.getBundled().getFilename()).isEqualTo(platformName + ".txt");
  }
}
