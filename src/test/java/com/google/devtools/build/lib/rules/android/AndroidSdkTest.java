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

import com.google.devtools.build.lib.rules.android.AndroidSdkTest.WithPlatforms;
import com.google.devtools.build.lib.rules.android.AndroidSdkTest.WithoutPlatforms;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Tests for {@link com.google.devtools.build.lib.rules.android.AndroidSdk}. */
@RunWith(Suite.class)
@SuiteClasses({WithoutPlatforms.class, WithPlatforms.class})
public abstract class AndroidSdkTest extends AndroidBuildViewTestCase {
  /** Use legacy toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithoutPlatforms extends AndroidSdkTest {}

  /** Use platform-based toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithPlatforms extends AndroidSdkTest {
    @Override
    protected boolean platformBasedToolchains() {
      return true;
    }
  }

  @Test
  public void testSourcePropertiesProvided() throws Exception {
    scratch.file(
        "sdk/BUILD",
        "android_sdk(",
        "    name = 'sdk',",
        "    aapt = 'static_aapt_tool',",
        "    adb = 'static_adb_tool',",
        "    aidl = 'static_aidl_tool',",
        "    framework_aidl = 'framework_aidl',",
        "    android_jar = 'android.jar',",
        "    source_properties = 'platforms/android-25/source.properties',",
        "    apksigner = 'apksigner',",
        "    dx = 'dx',",
        "    main_dex_classes = 'mainDexClasses.rules',",
        "    main_dex_list_creator = 'main_dex_list_creator',",
        "    proguard = 'ProGuard',",
        "    shrinked_android_jar = 'android.jar',",
        "    zipalign = 'zipalign',",
        "    tags = ['__ANDROID_RULES_MIGRATION__'],",
        ")");
    AndroidSdkProvider sdkProvider = getConfiguredTarget("//sdk").get(AndroidSdkProvider.PROVIDER);
    assertThat(sdkProvider.getSourceProperties().toDetailString())
        .isEqualTo("[/workspace[source]]sdk/platforms/android-25/source.properties");
  }
}
