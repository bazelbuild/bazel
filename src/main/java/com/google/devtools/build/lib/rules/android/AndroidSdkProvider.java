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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import javax.annotation.Nullable;

/** Description of the tools Blaze needs from an Android SDK. */
@AutoValue
@Immutable
public abstract class AndroidSdkProvider implements TransitiveInfoProvider {

  public static AndroidSdkProvider create(
      String buildToolsVersion,
      boolean aaptSupportsMainDexGeneration,
      Artifact frameworkAidl,
      @Nullable TransitiveInfoCollection aidlLib,
      Artifact androidJar,
      Artifact shrinkedAndroidJar,
      NestedSet<Artifact> androidBaseClasspathForJack,
      NestedSet<Artifact> javaBaseClasspathForJack,
      Artifact annotationsJar,
      Artifact mainDexClasses,
      FilesToRunProvider adb,
      FilesToRunProvider dx,
      FilesToRunProvider mainDexListCreator,
      FilesToRunProvider aidl,
      FilesToRunProvider aapt,
      FilesToRunProvider apkBuilder,
      FilesToRunProvider apkSigner,
      FilesToRunProvider proguard,
      FilesToRunProvider zipalign,
      FilesToRunProvider jack,
      FilesToRunProvider jill,
      FilesToRunProvider resourceExtractor) {

    return new AutoValue_AndroidSdkProvider(
        buildToolsVersion,
        aaptSupportsMainDexGeneration,
        frameworkAidl,
        aidlLib,
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
        resourceExtractor);
  }

  /**
   * Returns the Android SDK associated with the rule being analyzed or null if the Android SDK is
   * not specified.
   */
  public static AndroidSdkProvider fromRuleContext(RuleContext ruleContext) {
    TransitiveInfoCollection androidSdkDep =
        ruleContext.getPrerequisite(":android_sdk", Mode.TARGET);
    AndroidSdkProvider androidSdk = androidSdkDep == null
        ? null : androidSdkDep.getProvider(AndroidSdkProvider.class);

    return androidSdk;
  }

  /**
   * Signals an error if the Android SDK cannot be found.
   */
  public static boolean verifyPresence(RuleContext ruleContext) {
    if (fromRuleContext(ruleContext) == null) {
      ruleContext.ruleError(
          "No Android SDK found. Use the --android_sdk command line option to specify one.");
      return false;
    }

    return true;
  }

  /** The value of build_tools_version. May be null or empty. */
  public abstract String getBuildToolsVersion();

  public abstract boolean getAaptSupportsMainDexGeneration();

  public abstract Artifact getFrameworkAidl();

  @Nullable
  public abstract TransitiveInfoCollection getAidlLib();

  public abstract Artifact getAndroidJar();

  public abstract Artifact getShrinkedAndroidJar();

  /**
   * Returns the set of jack files to be used as a base classpath for jack compilation of Android
   * rules, typically a Jack translation of the jar returned by {@link getAndroidJar}.
   */
  public abstract NestedSet<Artifact> getAndroidBaseClasspathForJack();

  /**
   * Returns the set of jack files to be used as a base classpath for jack compilation of Java
   * rules, typically a Jack translation of the jars in the Java bootclasspath.
   */
  public abstract NestedSet<Artifact> getJavaBaseClasspathForJack();

  public abstract Artifact getAnnotationsJar();

  public abstract Artifact getMainDexClasses();

  public abstract FilesToRunProvider getAdb();

  public abstract FilesToRunProvider getDx();

  public abstract FilesToRunProvider getMainDexListCreator();

  public abstract FilesToRunProvider getAidl();

  public abstract FilesToRunProvider getAapt();

  public abstract FilesToRunProvider getApkBuilder();

  public abstract FilesToRunProvider getApkSigner();

  public abstract FilesToRunProvider getProguard();

  public abstract FilesToRunProvider getZipalign();

  public abstract FilesToRunProvider getJack();

  public abstract FilesToRunProvider getJill();

  public abstract FilesToRunProvider getResourceExtractor();

  AndroidSdkProvider() {}
}
