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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;

/** Implementation of the {@code android_sdk} rule. */
public class AndroidSdkBase implements RuleConfiguredTargetFactory {

  private final AndroidSemantics androidSemantics;

  protected AndroidSdkBase(AndroidSemantics androidSemantics) {
    this.androidSemantics = androidSemantics;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    androidSemantics.checkForMigrationTag(ruleContext);

    // If the user didn't specify --proguard_top, go with the proguard attribute in the android_sdk
    // rule. Otherwise, use what they told us to.
    FilesToRunProvider proguard =
        ruleContext.getFragment(JavaConfiguration.class).getProguardBinary() == null
            ? ruleContext.getExecutablePrerequisite("proguard", TransitionMode.HOST)
            : ruleContext.getExecutablePrerequisite(":proguard", TransitionMode.HOST);

    String buildToolsVersion =
        AggregatingAttributeMapper.of(ruleContext.getRule())
            .get("build_tools_version", Type.STRING);
    FilesToRunProvider aidl = ruleContext.getExecutablePrerequisite("aidl", TransitionMode.HOST);
    FilesToRunProvider aapt = ruleContext.getExecutablePrerequisite("aapt", TransitionMode.HOST);
    FilesToRunProvider aapt2 = ruleContext.getExecutablePrerequisite("aapt2", TransitionMode.HOST);
    FilesToRunProvider apkBuilder =
        ruleContext.getExecutablePrerequisite("apkbuilder", TransitionMode.HOST);
    FilesToRunProvider apkSigner =
        ruleContext.getExecutablePrerequisite("apksigner", TransitionMode.HOST);

    FilesToRunProvider adb = ruleContext.getExecutablePrerequisite("adb", TransitionMode.HOST);
    FilesToRunProvider dx = ruleContext.getExecutablePrerequisite("dx", TransitionMode.HOST);
    FilesToRunProvider mainDexListCreator =
        ruleContext.getExecutablePrerequisite("main_dex_list_creator", TransitionMode.HOST);
    FilesToRunProvider zipalign =
        ruleContext.getExecutablePrerequisite("zipalign", TransitionMode.HOST);
    Artifact frameworkAidl =
        ruleContext.getPrerequisiteArtifact("framework_aidl", TransitionMode.HOST);
    TransitiveInfoCollection aidlLib =
        ruleContext.getPrerequisite("aidl_lib", TransitionMode.TARGET);
    Artifact androidJar = ruleContext.getPrerequisiteArtifact("android_jar", TransitionMode.HOST);
    Artifact sourceProperties = ruleContext.getHostPrerequisiteArtifact("source_properties");
    Artifact shrinkedAndroidJar =
        ruleContext.getPrerequisiteArtifact("shrinked_android_jar", TransitionMode.HOST);
    Artifact mainDexClasses =
        ruleContext.getPrerequisiteArtifact("main_dex_classes", TransitionMode.HOST);

    if (ruleContext.hasErrors()) {
      return null;
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addNativeDeclaredProvider(
            new AndroidSdkProvider(
                buildToolsVersion,
                frameworkAidl,
                aidlLib,
                androidJar,
                sourceProperties,
                shrinkedAndroidJar,
                mainDexClasses,
                adb,
                dx,
                mainDexListCreator,
                aidl,
                aapt,
                aapt2,
                apkBuilder,
                apkSigner,
                proguard,
                zipalign,
                /* system= */ null))
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .setFilesToBuild(NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER))
        .build();
  }
}
