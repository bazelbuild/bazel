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
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import javax.annotation.Nullable;

/** Description of the tools Blaze needs from an Android SDK. */
@AutoValue
@Immutable
public abstract class AndroidSdkProvider implements TransitiveInfoProvider {

  public static AndroidSdkProvider create(
      String buildToolsVersion,
      Artifact frameworkAidl,
      @Nullable TransitiveInfoCollection aidlLib,
      Artifact androidJar,
      @Nullable Artifact sourceProperties,
      Artifact shrinkedAndroidJar,
      Artifact annotationsJar,
      Artifact mainDexClasses,
      FilesToRunProvider adb,
      FilesToRunProvider dx,
      FilesToRunProvider mainDexListCreator,
      FilesToRunProvider aidl,
      FilesToRunProvider aapt,
      @Nullable FilesToRunProvider aapt2,
      @Nullable FilesToRunProvider apkBuilder,
      FilesToRunProvider apkSigner,
      FilesToRunProvider proguard,
      FilesToRunProvider zipalign) {

    return new AutoValue_AndroidSdkProvider(
        buildToolsVersion,
        frameworkAidl,
        aidlLib,
        androidJar,
        sourceProperties,
        shrinkedAndroidJar,
        annotationsJar,
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
        zipalign);
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
   * Throws an error if the Android SDK cannot be found.
   */
  public static void verifyPresence(RuleContext ruleContext) throws RuleErrorException {
    if (fromRuleContext(ruleContext) == null) {
      throw ruleContext.throwWithRuleError(
          "No Android SDK found. Use the --android_sdk command line option to specify one.");
    }
  }

  /** The value of build_tools_version. May be null or empty. */
  public abstract String getBuildToolsVersion();

  public abstract Artifact getFrameworkAidl();

  @Nullable
  public abstract TransitiveInfoCollection getAidlLib();

  public abstract Artifact getAndroidJar();

  @Nullable
  public abstract Artifact getSourceProperties();

  public abstract Artifact getShrinkedAndroidJar();

  public abstract Artifact getAnnotationsJar();

  public abstract Artifact getMainDexClasses();

  public abstract FilesToRunProvider getAdb();

  public abstract FilesToRunProvider getDx();

  public abstract FilesToRunProvider getMainDexListCreator();

  public abstract FilesToRunProvider getAidl();

  public abstract FilesToRunProvider getAapt();

  @Nullable
  public abstract FilesToRunProvider getAapt2();

  @Nullable
  public abstract FilesToRunProvider getApkBuilder();

  public abstract FilesToRunProvider getApkSigner();

  public abstract FilesToRunProvider getProguard();

  public abstract FilesToRunProvider getZipalign();

  AndroidSdkProvider() {}
}
