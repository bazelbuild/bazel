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

import static com.google.devtools.build.lib.rules.android.AndroidStarlarkData.fromNoneable;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.java.BootClassPathInfo;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidSdkProviderApi;
import com.google.devtools.build.lib.syntax.EvalException;
import javax.annotation.Nullable;

/** Description of the tools Blaze needs from an Android SDK. */
@Immutable
public final class AndroidSdkProvider extends NativeInfo
    implements AndroidSdkProviderApi<Artifact, FilesToRunProvider, TransitiveInfoCollection> {

  public static final String ANDROID_SDK_TOOLCHAIN_TYPE_ATTRIBUTE_NAME =
      "$android_sdk_toolchain_type";

  public static final Provider PROVIDER = new Provider();

  private final String buildToolsVersion;
  private final Artifact frameworkAidl;
  private final TransitiveInfoCollection aidlLib;
  private final Artifact androidJar;
  private final Artifact sourceProperties;
  private final Artifact shrinkedAndroidJar;
  private final Artifact mainDexClasses;
  private final FilesToRunProvider adb;
  private final FilesToRunProvider dx;
  private final FilesToRunProvider mainDexListCreator;
  private final FilesToRunProvider aidl;
  private final FilesToRunProvider aapt;
  private final FilesToRunProvider aapt2;
  private final FilesToRunProvider apkBuilder;
  private final FilesToRunProvider apkSigner;
  private final FilesToRunProvider proguard;
  private final FilesToRunProvider zipalign;
  @Nullable private final BootClassPathInfo system;

  public AndroidSdkProvider(
      String buildToolsVersion,
      Artifact frameworkAidl,
      @Nullable TransitiveInfoCollection aidlLib,
      Artifact androidJar,
      @Nullable Artifact sourceProperties,
      Artifact shrinkedAndroidJar,
      Artifact mainDexClasses,
      FilesToRunProvider adb,
      FilesToRunProvider dx,
      FilesToRunProvider mainDexListCreator,
      FilesToRunProvider aidl,
      FilesToRunProvider aapt,
      FilesToRunProvider aapt2,
      @Nullable FilesToRunProvider apkBuilder,
      FilesToRunProvider apkSigner,
      FilesToRunProvider proguard,
      FilesToRunProvider zipalign,
      @Nullable BootClassPathInfo system) {
    super(PROVIDER);
    this.buildToolsVersion = buildToolsVersion;
    this.frameworkAidl = frameworkAidl;
    this.aidlLib = aidlLib;
    this.androidJar = androidJar;
    this.sourceProperties = sourceProperties;
    this.shrinkedAndroidJar = shrinkedAndroidJar;
    this.mainDexClasses = mainDexClasses;
    this.adb = adb;
    this.dx = dx;
    this.mainDexListCreator = mainDexListCreator;
    this.aidl = aidl;
    this.aapt = aapt;
    this.aapt2 = aapt2;
    this.apkBuilder = apkBuilder;
    this.apkSigner = apkSigner;
    this.proguard = proguard;
    this.zipalign = zipalign;
    this.system = system;
  }

  /**
   * Returns the Android SDK associated with the rule being analyzed or null if the Android SDK is
   * not specified.
   */
  public static AndroidSdkProvider fromRuleContext(RuleContext ruleContext) {
    return ruleContext.getPrerequisite(
        ":android_sdk", TransitionMode.TARGET, AndroidSdkProvider.PROVIDER);
  }

  /** Throws an error if the Android SDK cannot be found. */
  public static void verifyPresence(RuleContext ruleContext) throws RuleErrorException {
    if (fromRuleContext(ruleContext) == null) {
      throw ruleContext.throwWithRuleError(
          "No Android SDK found. Use the --android_sdk command line option to specify one.");
    }
  }

  @Override
  public String getBuildToolsVersion() {
    return buildToolsVersion;
  }

  @Override
  public Artifact getFrameworkAidl() {
    return frameworkAidl;
  }

  @Override
  @Nullable
  public TransitiveInfoCollection getAidlLib() {
    return aidlLib;
  }

  @Override
  public Artifact getAndroidJar() {
    return androidJar;
  }

  @Override
  @Nullable
  public Artifact getSourceProperties() {
    return sourceProperties;
  }

  @Override
  public Artifact getShrinkedAndroidJar() {
    return shrinkedAndroidJar;
  }

  @Override
  public Artifact getMainDexClasses() {
    return mainDexClasses;
  }

  @Override
  public FilesToRunProvider getAdb() {
    return adb;
  }

  @Override
  public FilesToRunProvider getDx() {
    return dx;
  }

  @Override
  public FilesToRunProvider getMainDexListCreator() {
    return mainDexListCreator;
  }

  @Override
  public FilesToRunProvider getAidl() {
    return aidl;
  }

  @Override
  public FilesToRunProvider getAapt() {
    return aapt;
  }

  @Override
  public FilesToRunProvider getAapt2() {
    return aapt2;
  }

  @Override
  @Nullable
  public FilesToRunProvider getApkBuilder() {
    return apkBuilder;
  }

  @Override
  public FilesToRunProvider getApkSigner() {
    return apkSigner;
  }

  @Override
  public FilesToRunProvider getProguard() {
    return proguard;
  }

  @Override
  public FilesToRunProvider getZipalign() {
    return zipalign;
  }

  public BootClassPathInfo getSystem() {
    return system;
  }

  /** The provider can construct the Android SDK provider. */
  public static class Provider extends BuiltinProvider<AndroidSdkProvider>
      implements AndroidSdkProviderApi.Provider<
          Artifact, FilesToRunProvider, TransitiveInfoCollection> {

    private Provider() {
      super(NAME, AndroidSdkProvider.class);
    }

    @Override
    public AndroidSdkProvider createInfo(
        String buildToolsVersion,
        Artifact frameworkAidl,
        Object aidlLib,
        Artifact androidJar,
        Object sourceProperties,
        Artifact shrinkedAndroidJar,
        Artifact mainDexClasses,
        FilesToRunProvider adb,
        FilesToRunProvider dx,
        FilesToRunProvider mainDexListCreator,
        FilesToRunProvider aidl,
        FilesToRunProvider aapt,
        FilesToRunProvider aapt2,
        Object apkBuilder,
        FilesToRunProvider apkSigner,
        FilesToRunProvider proguard,
        FilesToRunProvider zipalign,
        Object system)
        throws EvalException {
      return new AndroidSdkProvider(
          buildToolsVersion,
          frameworkAidl,
          fromNoneable(aidlLib, TransitiveInfoCollection.class),
          androidJar,
          fromNoneable(sourceProperties, Artifact.class),
          shrinkedAndroidJar,
          mainDexClasses,
          adb,
          dx,
          mainDexListCreator,
          aidl,
          aapt,
          aapt2,
          fromNoneable(apkBuilder, FilesToRunProvider.class),
          apkSigner,
          proguard,
          zipalign,
          fromNoneable(system, BootClassPathInfo.class));
    }
  }
}
