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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.MultidexMode;
import java.util.Set;
import org.junit.Before;

/**
 * Base class for testing the multidex code of android_binary and android_test.
 */
public class AndroidMultidexBaseTest extends BuildViewTestCase {

  @Before
  public final void createFiles() throws Exception {
    scratch.file("java/android/BUILD",
        "android_binary(name = 'app',",
        "               srcs = ['A.java'],",
        "               manifest = 'AndroidManifest.xml',",
        "               resource_files = glob(['res/**']),",
        "              )");
    scratch.file("java/android/res/values/strings.xml",
        "<resources><string name = 'hello'>Hello Android!</string></resources>");
    scratch.file("java/android/A.java",
        "package android; public class A {};");
  }

  /**
   * Internal helper method: given an android_binary rule label, check that it builds in
   * multidex mode. Three multidex variations are possible: "legacy", "manual_main_dex" and
   * "native".
   *
   * @param ruleLabel the android_binary rule label to test against
   * @param multidexMode the multidex mode used in the rule
   */
  protected void internalTestMultidexBuildStructure(
      String ruleLabel, MultidexMode multidexMode) throws Exception {

    ConfiguredTarget binary = getConfiguredTarget(ruleLabel);
    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));

    // Always created:
    Artifact intermediateDexOutput =
        getFirstArtifactEndingWith(artifacts, "intermediate_classes.dex.zip");
    assertThat(intermediateDexOutput).isNotNull();
    Artifact finalDexOutput = getFirstArtifactEndingWith(artifacts, "/classes.dex.zip");
    assertThat(finalDexOutput).isNotNull();

    // Only created in legacy mode:
    Artifact strippedJar = getFirstArtifactEndingWith(artifacts, "main_dex_intermediate.jar");
    Artifact mainDexList = getFirstArtifactEndingWith(artifacts, "main_dex_list.txt");

    if (multidexMode == MultidexMode.LEGACY) {
      // First action: check that the stripped jar is generated through Proguard.
      AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(getRuleContext(binary));
      assertThat(strippedJar).isNotNull();
      SpawnAction stripAction = (SpawnAction) getGeneratingAction(strippedJar);
      assertThat(stripAction.getCommandFilename())
          .isEqualTo(sdk.getProguard().getExecutable().getExecPathString());

      // Second action: The dexer consumes the stripped jar to create the main dex class list.
      assertThat(mainDexList).isNotNull();
      SpawnAction mainDexAction = (SpawnAction) getGeneratingAction(mainDexList);
      assertThat(mainDexAction.getArguments()).containsAllOf(
          mainDexList.getExecPathString(),
          strippedJar.getExecPathString()).inOrder();
    } else if (multidexMode == MultidexMode.MANUAL_MAIN_DEX) {
      // Manual main dex mode: strippedJar shouldn't exist and mainDexList should be provided.
      assertThat(strippedJar).isNull();
      assertThat(mainDexList).isNotNull();
    } else {
      // Native mode: these shouldn't exist.
      assertThat(strippedJar).isNull();
      assertThat(mainDexList).isNull();
    }

    // Third action: the dexer command consumes the main dex class list to generate the intermediate
    // dex output.
    SpawnAction dexAction = (SpawnAction) getGeneratingAction(intermediateDexOutput);
    ImmutableList.Builder<String> argBuilder = ImmutableList.builder();
    argBuilder.add("--dex");
    if (multidexMode == MultidexMode.OFF) {
      argBuilder.add("--num-threads=5");
    } else {
      argBuilder.add("--multi-dex");

      if (multidexMode == MultidexMode.LEGACY || multidexMode == MultidexMode.MANUAL_MAIN_DEX) {
        argBuilder.add("--main-dex-list=" + mainDexList.getExecPathString());
      }
    }

    argBuilder.add("--output=" + intermediateDexOutput.getExecPathString());

    assertThat(dexAction.getArguments()).containsAllIn(argBuilder.build()).inOrder();

    // Final action: the SingleJar command that consumes the intermediate dex out to produce the
    // final dex out.
    SpawnAction singleJarAction = getGeneratingSpawnAction(finalDexOutput);
    assertThat(singleJarAction.getArguments())
        .containsAllOf(
            "--exclude_build_data",
            "--dont_change_compression",
            "--sources",
            intermediateDexOutput.getExecPathString(),
            "--output",
            finalDexOutput.getExecPathString(),
            "--include_prefixes",
            "classes")
        .inOrder();
  }
  
  /**
   * Internal helper method: given an android_binary rule label, check that it builds
   * in non-multidex mode.
   */
  protected void internalTestNonMultidexBuildStructure(String ruleLabel) throws Exception {

    ConfiguredTarget binary = getConfiguredTarget(ruleLabel);
    Set<Artifact> artifacts = actionsTestUtil().artifactClosureOf(getFilesToBuild(binary));
    Artifact dexOutput = getFirstArtifactEndingWith(artifacts, "classes.dex");
    SpawnAction dexAction = (SpawnAction) getGeneratingAction(dexOutput);

    assertThat(dexAction.getArguments()).doesNotContain("--multi-dex");
    assertThat(dexAction.getArguments()).containsAllOf(
            "--dex",
            "--num-threads=5",
            "--output=" + dexOutput.getExecPathString()).inOrder();
  }
}

