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
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Description of the tools Blaze needs from an Android SDK.
 */
@Immutable
public final class AndroidSdkProvider implements TransitiveInfoProvider {

  private final String buildToolsVersion;
  private final Artifact frameworkAidl;
  private final Artifact androidJar;
  private final Artifact shrinkedAndroidJar;
  private final Artifact androidJack;
  private final Artifact annotationsJar;
  private final Artifact mainDexClasses;
  private final FilesToRunProvider adb;
  private final FilesToRunProvider dx;
  private final FilesToRunProvider mainDexListCreator;
  private final FilesToRunProvider aidl;
  private final FilesToRunProvider aapt;
  private final FilesToRunProvider apkBuilder;
  private final FilesToRunProvider proguard;
  private final FilesToRunProvider zipalign;
  private final FilesToRunProvider jack;
  private final FilesToRunProvider jill;
  private final FilesToRunProvider resourceExtractor;

  public AndroidSdkProvider(
      String buildToolsVersion,
      Artifact frameworkAidl,
      Artifact androidJar,
      Artifact shrinkedAndroidJar,
      Artifact androidJack,
      Artifact annotationsJar,
      Artifact mainDexClasses,
      FilesToRunProvider adb,
      FilesToRunProvider dx,
      FilesToRunProvider mainDexListCreator,
      FilesToRunProvider aidl,
      FilesToRunProvider aapt,
      FilesToRunProvider apkBuilder,
      FilesToRunProvider proguard,
      FilesToRunProvider zipalign,
      FilesToRunProvider jack,
      FilesToRunProvider jill,
      FilesToRunProvider resourceExtractor) {

    this.buildToolsVersion = buildToolsVersion;
    this.frameworkAidl = frameworkAidl;
    this.androidJar = androidJar;
    this.shrinkedAndroidJar = shrinkedAndroidJar;
    this.androidJack = androidJack;
    this.annotationsJar = annotationsJar;
    this.mainDexClasses = mainDexClasses;
    this.adb = adb;
    this.dx = dx;
    this.mainDexListCreator = mainDexListCreator;
    this.aidl = aidl;
    this.aapt = aapt;
    this.apkBuilder = apkBuilder;
    this.proguard = proguard;
    this.zipalign = zipalign;
    this.jack = jack;
    this.jill = jill;
    this.resourceExtractor = resourceExtractor;
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

  /**
   * The value of build_tools_version. May be null or empty.
   */
  public String getBuildToolsVersion() {
    return buildToolsVersion;
  }
  
  public Artifact getFrameworkAidl() {
    return frameworkAidl;
  }

  public Artifact getAndroidJar() {
    return androidJar;
  }

  public Artifact getShrinkedAndroidJar() {
    return shrinkedAndroidJar;
  }

  public Artifact getAndroidJack() {
    return androidJack;
  }

  public Artifact getAnnotationsJar() {
    return annotationsJar;
  }

  public Artifact getMainDexClasses() {
    return mainDexClasses;
  }

  public FilesToRunProvider getAdb() {
    return adb;
  }

  public FilesToRunProvider getDx() {
    return dx;
  }

  public FilesToRunProvider getMainDexListCreator() {
    return mainDexListCreator;
  }

  public FilesToRunProvider getAidl() {
    return aidl;
  }

  public FilesToRunProvider getAapt() {
    return aapt;
  }

  public FilesToRunProvider getApkBuilder() {
    return apkBuilder;
  }

  public FilesToRunProvider getProguard() {
    return proguard;
  }

  public FilesToRunProvider getZipalign() {
    return zipalign;
  }

  public FilesToRunProvider getJack() {
    return jack;
  }

  public FilesToRunProvider getJill() {
    return jill;
  }

  public FilesToRunProvider getResourceExtractor() {
    return resourceExtractor;
  }
}
