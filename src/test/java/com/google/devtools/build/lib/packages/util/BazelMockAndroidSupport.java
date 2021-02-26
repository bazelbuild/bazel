// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages.util;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.Pair;
import java.io.IOException;

/** Mocks out Android dependencies for testing. */
public final class BazelMockAndroidSupport {

  private BazelMockAndroidSupport() {}

  public static void setupNdk(MockToolsConfig config) throws IOException {
    new Crosstool(config, "android/crosstool", Label.parseAbsoluteUnchecked("//android/crosstool"))
        .setCcToolchainFile(
            ResourceLoader.readFromResources(
                "com/google/devtools/build/lib/packages/util/mock/android_cc_toolchain_config.bzl"))
        .setToolchainConfigs(ImmutableList.of(x86Config().build(), armeabiV7a().build()))
        .setSupportedArchs(ImmutableList.of("x86", "armeabi-v7a"))
        .setSupportsHeaderParsing(false)
        .write();
  }

  public static CcToolchainConfig.Builder x86Config() {
    return CcToolchainConfig.builder()
        .withCpu("x86")
        .withCompiler("gcc")
        .withToolchainIdentifier("x86")
        .withHostSystemName("x86")
        .withTargetSystemName("x86-linux-android")
        .withTargetLibc("local")
        .withAbiVersion("x86")
        .withAbiLibcVersion("r7")
        .withSysroot("")
        .withToolPaths(
            Pair.of("gcc", "x86/bin/i686-linux-android-gcc"),
            Pair.of("ar", "x86/bin/i686-linux-android-ar"),
            Pair.of("cpp", "x86/bin/i686-linux-android-cpp"),
            Pair.of("gcov", "x86/bin/i686-linux-android-gcov"),
            Pair.of("ld", "x86/bin/i686-linux-android-ld"),
            Pair.of("nm", "x86/bin/i686-linux-android-nm"),
            Pair.of("objcopy", "x86/bin/i686-linux-android-objcopy"),
            Pair.of("objdump", "x86/bin/i686-linux-android-objdump"),
            Pair.of("strip", "x86/bin/i686-linux-android-strip"),
            Pair.of("ld-bfd", "x86/bin/i686-linux-android-ld.bfd"),
            Pair.of("ld-gold", "x86/bin/i686-linux-android-ld.gold"));
  }

  public static CcToolchainConfig.Builder armeabiV7a() {
    return CcToolchainConfig.builder()
        .withCpu("armeabi-v7a")
        .withCompiler("gcc")
        .withToolchainIdentifier("armeabi-v7a")
        .withHostSystemName("x86")
        .withTargetSystemName("arm-linux-androideabi")
        .withTargetLibc("local")
        .withAbiVersion("armeabi-v7a")
        .withAbiLibcVersion("r7")
        .withSysroot("")
        .withToolPaths(
            Pair.of("gcc", "arm/bin/arm-linux-androideabi-gcc"),
            Pair.of("ar", "arm/bin/arm-linux-androideabi-ar"),
            Pair.of("cpp", "arm/bin/arm-linux-androideabi-cpp"),
            Pair.of("gcov", "arm/bin/arm-linux-androideabi-gcov"),
            Pair.of("ld", "arm/bin/arm-linux-androideabi-ld"),
            Pair.of("nm", "arm/bin/arm-linux-androideabi-nm"),
            Pair.of("objcopy", "arm/bin/arm-linux-androideabi-objcopy"),
            Pair.of("objdump", "arm/bin/arm-linux-androideabi-objdump"),
            Pair.of("strip", "arm/bin/arm-linux-androideabi-strip"),
            Pair.of("ld-bfd", "arm/bin/arm-linux-androideabi-ld.bfd"),
            Pair.of("ld-gold", "arm/bin/arm-linux-androideabi-ld.gold"));
  }

  public static void setupPlatformResolvableSdks(MockToolsConfig config) throws Exception {
    config.create(
        "platform_selected_android_sdks/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "android_sdk(",
        "    name = 'x86_64',",
        "    aapt = ':aapt_x86_64',",
        "    aapt2 = ':aapt2_x86_64',",
        "    adb = ':adb_x86_64',",
        "    aidl = ':aidl_x86_64',",
        "    android_jar = ':android_x86_64.jar',",
        "    apksigner = ':apksigner_x86_64',",
        "    dx = ':dx_x86_64',",
        "    framework_aidl = ':framework_idl_x86_64',",
        "    main_dex_classes = ':main_dex_classes_x86_64',",
        "    main_dex_list_creator = ':main_dex_list_creator_x86_64',",
        "    proguard = ':proguard_x86_64',",
        "    shrinked_android_jar =':shrinked_android_x86_64.jar',",
        "    zipalign = ':zipalign_x86_64',",
        "    tags = ['__ANDROID_RULES_MIGRATION__'],",
        ")",
        "android_sdk(",
        "    name = 'arm',",
        "    aapt = ':aapt_arm',",
        "    aapt2 = ':aapt2_arm',",
        "    adb = ':adb_arm',",
        "    aidl = ':aidl_arm',",
        "    android_jar = ':android_arm.jar',",
        "    apksigner = ':apksigner_arm',",
        "    dx = ':dx_arm',",
        "    framework_aidl = ':framework_idl_arm',",
        "    main_dex_classes = ':main_dex_classes_arm',",
        "    main_dex_list_creator = ':main_dex_list_creator_arm',",
        "    proguard = ':proguard_arm',",
        "    shrinked_android_jar =':shrinked_android_arm.jar',",
        "    zipalign = ':zipalign_arm',",
        "    tags = ['__ANDROID_RULES_MIGRATION__'],",
        ")");

    config.create(
        "platform_selected_android_sdks/toolchains/BUILD",
        "toolchain(",
        "    name = 'x86_64_toolchain',",
        String.format("    toolchain_type = '%s',", TestConstants.ANDROID_TOOLCHAIN_TYPE_LABEL),
        "    toolchain = '//platform_selected_android_sdks:x86_64',",
        "    target_compatible_with = [",
        "        '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_64',",
        "    ])",
        "toolchain(",
        "    name = 'arm_toolchain',",
        String.format("    toolchain_type = '%s',", TestConstants.ANDROID_TOOLCHAIN_TYPE_LABEL),
        "    toolchain = '//platform_selected_android_sdks:arm',",
        "    target_compatible_with = [",
        "        '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm',",
        "    ])");
  }
}
