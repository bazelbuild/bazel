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
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.util.Pair;
import java.io.IOException;

/** Mocks out Android dependencies for testing. */
public final class BazelMockAndroidSupport {

  private BazelMockAndroidSupport() {}

  public static void setupNdk(MockToolsConfig config) throws IOException {
    new Crosstool(config, "android/crosstool")
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
}
