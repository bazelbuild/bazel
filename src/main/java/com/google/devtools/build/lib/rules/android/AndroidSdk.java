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

import com.android.repository.Revision;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.java.BaseJavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.syntax.Type;
import java.util.Collection;

/**
 * Implementation of the {@code android_sdk} rule.
 */
public class AndroidSdk implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    // If the user didn't specify --proguard_top, go with the proguard attribute in the android_sdk
    // rule. Otherwise, use what she told us to.
    FilesToRunProvider proguard =
        ruleContext.getFragment(JavaConfiguration.class).getProguardBinary() == null
            ? ruleContext.getExecutablePrerequisite("proguard", Mode.HOST)
            : ruleContext.getExecutablePrerequisite(":proguard", Mode.HOST);

    String buildToolsVersion = AggregatingAttributeMapper.of(ruleContext.getRule())
        .get("build_tools_version", Type.STRING);
    Revision parsedBuildToolsVersion = null;
    try {
      parsedBuildToolsVersion =
          Strings.isNullOrEmpty(buildToolsVersion)
              ? null
              : Revision.parseRevision(buildToolsVersion);
    } catch (NumberFormatException nfe) {
      ruleContext.attributeError("build_tools_version", "Invalid version: " + buildToolsVersion);
    }
    boolean aaptSupportsMainDexGeneration =
        parsedBuildToolsVersion == null
            || parsedBuildToolsVersion.compareTo(new Revision(24)) >= 0;
    FilesToRunProvider aidl = ruleContext.getExecutablePrerequisite("aidl", Mode.HOST);
    FilesToRunProvider aapt = ruleContext.getExecutablePrerequisite("aapt", Mode.HOST);
    FilesToRunProvider apkBuilder = ruleContext.getExecutablePrerequisite(
        "apkbuilder", Mode.HOST);
    FilesToRunProvider apkSigner = ruleContext.getExecutablePrerequisite("apksigner", Mode.HOST);

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
    // Because all Jack actions using this android_sdk will need Jack versions of the Android and
    // Java classpaths, pre-translate the jars for Android and Java targets here. (They will only
    // be run if needed, as usual for Bazel.)
    NestedSet<Artifact> androidBaseClasspathForJack =
        convertClasspathJarsToJack(
            ruleContext, jack, jill, resourceExtractor, ImmutableList.of(androidJar));
    NestedSet<Artifact> javaBaseClasspathForJack =
        convertClasspathJarsToJack(
            ruleContext,
            jack,
            jill,
            resourceExtractor,
            BaseJavaCompilationHelper.getBootClasspath(
                ruleContext, JavaToolchainProvider.fromRuleContext(ruleContext), ""));
    Artifact annotationsJar = ruleContext.getPrerequisiteArtifact("annotations_jar", Mode.HOST);
    Artifact mainDexClasses = ruleContext.getPrerequisiteArtifact("main_dex_classes", Mode.HOST);

    if (ruleContext.hasErrors()) {
      return null;
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .add(
            AndroidSdkProvider.class,
            AndroidSdkProvider.create(
                buildToolsVersion,
                aaptSupportsMainDexGeneration,
                frameworkAidl,
                androidJar,
                shrinkedAndroidJar,
                androidBaseClasspathForJack,
                javaBaseClasspathForJack,
                annotationsJar,
                mainDexClasses,
                adb,
                dx,
                mainDexListCreator,
                aidl,
                aapt,
                apkBuilder,
                apkSigner,
                proguard,
                zipalign,
                jack,
                jill,
                resourceExtractor))
        .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .setFilesToBuild(NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER))
        .build();
  }

  private NestedSet<Artifact> convertClasspathJarsToJack(
      RuleContext ruleContext,
      FilesToRunProvider jack,
      FilesToRunProvider jill,
      FilesToRunProvider resourceExtractor,
      Collection<Artifact> jars) {
    return new JackCompilationHelper.Builder()
        // bazel infrastructure
        .setRuleContext(ruleContext)
        // configuration
        .setTolerant()
        // tools
        .setJackBinary(jack)
        .setJillBinary(jill)
        .setResourceExtractorBinary(resourceExtractor)
        .setJackBaseClasspath(NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER))
        // sources
        .addCompiledJars(jars)
        .build()
        .compileAsLibrary()
        .getTransitiveJackClasspathLibraries();
  }
}
