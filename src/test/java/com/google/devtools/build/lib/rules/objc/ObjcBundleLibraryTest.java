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
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.NESTED_BUNDLE;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.testutil.Scratch;
import java.io.IOException;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for objc_bundle_library. */
@RunWith(JUnit4.class)
public class ObjcBundleLibraryTest extends ObjcRuleTestCase {
  protected static final RuleType RULE_TYPE =
      new RuleType("objc_bundle_library") {
        @Override
        Iterable<String> requiredAttributes(
            Scratch scratch, String packageDir, Set<String> alreadyAdded) {
          return ImmutableList.of();
        }
      };

  private void addBundleWithResource() throws IOException {
    scratch.file("bndl/foo.data");
    scratch.file("bndl/BUILD",
        "objc_bundle_library(",
        "    name = 'bndl',",
        "    resources = ['foo.data'],",
        ")");
  }

  @Test
  public void testDoesNotGenerateLinkActionWhenThereAreNoSources() throws Exception {
    addBundleWithResource();
    assertThat(linkAction("//bndl:bndl")).isNull();
    ObjcProvider objcProvider = providerForTarget("//bndl:bndl");
    Bundling providedBundle =
        Iterables.getOnlyElement(objcProvider.get(NESTED_BUNDLE));
    assertThat(providedBundle.getCombinedArchitectureBinary()).isAbsent();
  }

  @Test
  public void testCreate_actoolAction() throws Exception {
    addTargetWithAssetCatalogs(RULE_TYPE);
    checkActoolActionCorrectness(DEFAULT_IOS_SDK_VERSION);
  }

  @Test
  public void testRegistersStoryboardCompilationActions() throws Exception {
    checkRegistersStoryboardCompileActions(RULE_TYPE, "iphone");
  }

  @Test
  public void testCompileXibActions() throws Exception {
    checkCompileXibActions(RULE_TYPE);
  }

  @Test
  public void testTwoStringsOneBundlePath() throws Exception {
    checkTwoStringsOneBundlePath(RULE_TYPE);
  }

  @Test
  public void testTwoResourcesOneBundlePath() throws Exception {
    checkTwoResourcesOneBundlePath(RULE_TYPE);
  }

  @Test
  public void testSameStringTwice() throws Exception {
    checkSameStringsTwice(RULE_TYPE);
  }

  @Test
  public void testPassesFamiliesToIbtool() throws Exception {
    checkPassesFamiliesToIbtool(RULE_TYPE);
  }

  @Test
  public void testMultipleInfoPlists() throws Exception {
    checkMultipleInfoPlists(RULE_TYPE);
  }

  @Test
  public void testInfoplistAndInfoplistsTogether() throws Exception {
    checkInfoplistAndInfoplistsTogether(RULE_TYPE);
  }

  @Test
  // Regression test for b/34770913.
  public void testDeviceAndSimulatorBuilds() throws Exception {
    addBundleWithResource();
    useConfiguration("--ios_multi_cpus=i386,x86_64,arm64,armv7");

    // Verifies this rule does not raise a rule error based on bundling platform flags.
    providerForTarget("//bndl:bndl");
  }
}
