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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos;
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
  public void testProvidesBundling() throws Exception {
    addBundleWithResource();
    scratch.file(
        "bin/BUILD",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    bundles = ['//bndl:bndl'],",
        ")");
    BundleMergeProtos.Control mergeControl = bundleMergeControl("//bin:bin");
    BundleMergeProtos.Control nestedControl =
        Iterables.getOnlyElement(mergeControl.getNestedBundleList());
    BundleMergeProtos.BundleFile bundleFile =
        BundleMergeProtos.BundleFile.newBuilder()
            .setBundlePath("foo.data")
            .setSourceFile(getSourceArtifact("bndl/foo.data").getExecPathString())
            .setExternalFileAttribute(BundleableFile.DEFAULT_EXTERNAL_FILE_ATTRIBUTE)
            .build();

    // Should put resource in .bundle directory, not bundle root (.app dir)
    assertThat(nestedControl.getBundleFileList()).containsExactly(bundleFile);
    assertThat(mergeControl.getBundleFileList()).doesNotContain(bundleFile);
  }

  @Test
  public void testDoesNotMergeInfoplistOfNestedBundle() throws Exception {
    scratch.file("bndl1/bndl1-Info.plist");
    scratch.file("bndl1/BUILD",
        "objc_bundle_library(",
        "    name = 'bndl1',",
        "    infoplist = 'bndl1-Info.plist',",
        ")");
    scratch.file("bndl2/bndl2-Info.plist");
    scratch.file("bndl2/BUILD",
        "objc_bundle_library(",
        "    name = 'bndl2',",
        "    bundles = ['//bndl1:bndl1'],",
        "    infoplist = 'bndl2-Info.plist',",
        ")");
    scratch.file("bin/bin-Info.plist");
    scratch.file("bin/bin.m");
    scratch.file("bin/BUILD",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = ['bin.m'],",
        "    bundles = ['//bndl2:bndl2'],",
        "    infoplist = 'bin-Info.plist'",
        ")");
    Artifact bundle1Infoplist = getSourceArtifact("bndl1/bndl1-Info.plist");
    Artifact bundle2Infoplist = getSourceArtifact("bndl2/bndl2-Info.plist");
    Artifact binaryInfoplist = getSourceArtifact("bin/bin-Info.plist");
    Artifact binaryMergedInfoplist = getMergedInfoPlist(getConfiguredTarget("//bin:bin"));

    PlMergeProtos.Control binaryPlMergeControl = plMergeControl("//bin:bin");

    assertThat(binaryPlMergeControl.getSourceFileList())
        .contains(binaryInfoplist.getExecPathString());
    assertThat(binaryPlMergeControl.getSourceFileList())
        .containsNoneOf(bundle1Infoplist.getExecPathString(), bundle2Infoplist.getExecPathString());

    assertThat(bundleMergeAction("//bin:bin").getInputs())
        .containsAllOf(bundle1Infoplist, bundle2Infoplist, binaryMergedInfoplist);

    BundleMergeProtos.Control binControl = bundleMergeControl("//bin:bin");
    assertThat(binControl.getBundleInfoPlistFile())
        .isEqualTo(binaryMergedInfoplist.getExecPathString());

    BundleMergeProtos.Control bundle2Control =
        Iterables.getOnlyElement(binControl.getNestedBundleList());
    assertThat(bundle2Control.getBundleInfoPlistFile())
        .isEqualTo(bundle2Infoplist.getExecPathString());

    BundleMergeProtos.Control bundle1Control =
        Iterables.getOnlyElement(bundle2Control.getNestedBundleList());
    assertThat(bundle1Control.getBundleInfoPlistFile())
        .isEqualTo(bundle1Infoplist.getExecPathString());
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
