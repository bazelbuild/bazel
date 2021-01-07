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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.ResolvedToolchainContext;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.java.BootClassPathInfo;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidSdkProviderApi;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** Description of the tools Blaze needs from an Android SDK. */
@Immutable
public final class AndroidSdkProvider extends NativeInfo
    implements AndroidSdkProviderApi<Artifact, FilesToRunProvider, TransitiveInfoCollection> {

  public static final String ANDROID_SDK_TOOLCHAIN_TYPE_ATTRIBUTE_NAME =
      "$android_sdk_toolchain_type";
  public static final String ANDROID_SDK_DUMMY_TOOLCHAIN_ATTRIBUTE_NAME =
      "$android_sdk_dummy_toolchains";

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

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  /**
   * Returns the Android SDK associated with the rule being analyzed or null if the Android SDK is
   * not specified.
   *
   * <p>First tries to read from toolchains if
   * --incompatible_enable_android_toolchain_resolution=true, else, uses the legacy attribute..
   */
  public static AndroidSdkProvider fromRuleContext(RuleContext ruleContext) {
    BuildConfiguration configuration = ruleContext.getConfiguration();
    if (configuration != null
        && configuration.hasFragment(AndroidConfiguration.class)
        && configuration
            .getFragment(AndroidConfiguration.class)
            .incompatibleUseToolchainResolution()) {
      AttributeMap attributes = ruleContext.attributes();
      if (ruleContext.getToolchainContext() == null) {
        ruleContext.ruleError(
            String.format(
                "'%s' rule '%s' requested sdk toolchain resolution via"
                    + " --incompatible_enable_android_toolchain_resolution but doesn't use"
                    + " toolchain resolution.",
                ruleContext.getRuleClassNameForLogging(), ruleContext.getLabel()));
        return null;
      }
      Label toolchainType =
          attributes.get(ANDROID_SDK_TOOLCHAIN_TYPE_ATTRIBUTE_NAME, BuildType.NODEP_LABEL);
      if (toolchainType == null) {
        ruleContext.ruleError(
            String.format(
                "'%s' rule '%s' requested sdk toolchain resolution via"
                    + " --incompatible_enable_android_toolchain_resolution but doesn't have"
                    + " toolchain type attribute '%s'.",
                ruleContext.getRuleClassNameForLogging(),
                ruleContext.getLabel(),
                ANDROID_SDK_TOOLCHAIN_TYPE_ATTRIBUTE_NAME));
        return null;
      }
      ResolvedToolchainContext toolchainContext = ruleContext.getToolchainContext();
      if (attributes.has(ANDROID_SDK_DUMMY_TOOLCHAIN_ATTRIBUTE_NAME, BuildType.NODEP_LABEL_LIST)) {
        ImmutableSet<Label> resolvedToolchains = toolchainContext.resolvedToolchainLabels();
        List<Label> dummyToochains =
            attributes.get(ANDROID_SDK_DUMMY_TOOLCHAIN_ATTRIBUTE_NAME, BuildType.NODEP_LABEL_LIST);
        for (Label toolchain : resolvedToolchains) {
          if (dummyToochains.contains(toolchain)) {
            ruleContext.ruleError(
                String.format(
                    "'%s' rule '%s' requested sdk toolchain resolution via"
                        + " --incompatible_enable_android_toolchain_resolution but hasn't set an"
                        + " appropriate --platforms value: --platforms=%s",
                    ruleContext.getRuleClassNameForLogging(),
                    ruleContext.getLabel(),
                    configuration.getFragment(PlatformConfiguration.class).getTargetPlatform()));
            return null;
          }
        }
      }
      ToolchainInfo info = toolchainContext.forToolchainType(toolchainType);
      if (info == null) {
        ruleContext.ruleError(
            String.format(
                "'%s' rule '%s' requested sdk toolchain resolution via"
                    + " --incompatible_enable_android_toolchain_resolution but doesn't have a"
                    + " toolchain for '%s'.",
                ruleContext.getRuleClassNameForLogging(), ruleContext.getLabel(), toolchainType));
        return null;
      }
      try {
        return (AndroidSdkProvider) info.getValue("android_sdk_info");
      } catch (EvalException e) {
        ruleContext.ruleError(
            String.format(
                "Android SDK toolchain for %s didn't have an 'android_sdk_info' provider: %s",
                ruleContext.getLabel(), e.getMessage()));
        return null;
      }
    }

    return ruleContext.getPrerequisite(":android_sdk", AndroidSdkProvider.PROVIDER);
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
