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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.syntax.Type;

/**
 * Implementation of the {@code android_sdk} rule.
 */
public class AndroidSdk implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    // If the user didn't specify --proguard_top, go with the proguard attribute in the android_sdk
    // rule. Otherwise, use what she told us to.
    FilesToRunProvider proguard =
        ruleContext.getFragment(JavaConfiguration.class).getProguardBinary() == null
            ? ruleContext.getExecutablePrerequisite("proguard", Mode.HOST)
            : ruleContext.getExecutablePrerequisite(":proguard", Mode.HOST);

    String buildToolsVersion = AggregatingAttributeMapper.of(ruleContext.getRule())
        .get("build_tools_version", Type.STRING);
    FilesToRunProvider aidl = ruleContext.getExecutablePrerequisite("aidl", Mode.HOST);
    FilesToRunProvider aapt = ruleContext.getExecutablePrerequisite("aapt", Mode.HOST);
    FilesToRunProvider apkBuilder = ruleContext.getExecutablePrerequisite(
        "apkbuilder", Mode.HOST);
    FilesToRunProvider adb = ruleContext.getExecutablePrerequisite("adb", Mode.HOST);
    FilesToRunProvider dx = ruleContext.getExecutablePrerequisite("dx", Mode.HOST);
    FilesToRunProvider mainDexListCreator = ruleContext.getExecutablePrerequisite(
        "main_dex_list_creator", Mode.HOST);
    FilesToRunProvider zipalign = ruleContext.getExecutablePrerequisite("zipalign", Mode.HOST);
    FilesToRunProvider jack = ruleContext.getExecutablePrerequisite("jack", Mode.HOST);
    FilesToRunProvider jill = ruleContext.getExecutablePrerequisite("jill", Mode.HOST);
    FilesToRunProvider resourceExtractor =
        ruleContext.getExecutablePrerequisite("resource_extractor", Mode.HOST);
    Artifact frameworkAidl = ruleContext.getPrerequisiteArtifact("framework_aidl", Mode.HOST);
    Artifact androidJar = ruleContext.getPrerequisiteArtifact("android_jar", Mode.HOST);
    Artifact shrinkedAndroidJar =
        ruleContext.getPrerequisiteArtifact("shrinked_android_jar", Mode.HOST);
    Artifact androidJack = ruleContext.getPrerequisiteArtifact("android_jack", Mode.HOST);
    Artifact annotationsJar = ruleContext.getPrerequisiteArtifact("annotations_jar", Mode.HOST);
    Artifact mainDexClasses = ruleContext.getPrerequisiteArtifact("main_dex_classes", Mode.HOST);

    if (ruleContext.hasErrors()) {
      return null;
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .add(
            AndroidSdkProvider.class,
            new AndroidSdkProvider(
                buildToolsVersion,
                frameworkAidl,
                androidJar,
                shrinkedAndroidJar,
                androidJack,
                annotationsJar,
                mainDexClasses,
                adb,
                dx,
                mainDexListCreator,
                aidl,
                aapt,
                apkBuilder,
                proguard,
                zipalign,
                jack,
                jill,
                resourceExtractor))
        .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .setFilesToBuild(NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER))
        .build();
  }
}
