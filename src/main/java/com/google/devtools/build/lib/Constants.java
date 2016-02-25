// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

/**
 * A temporary class of constants; these encode differences between Google's internal setup and
 * Bazel. We're working to remove this class, which requires cleaning up our internal code base.
 * Please don't add anything here unless you know what you're doing.
 *
 * <p>The extra {@code .toString()} calls are there so that javac doesn't inline these constants
 * so that we can replace this class file within a binary.
 */
public final class Constants {
  private Constants() {}

  // Google's internal name for Bazel is 'Blaze', and it will take some more time to change that.
  public static final String PRODUCT_NAME = "bazel";

  // Default value for the --package_path flag if not otherwise set.
  public static final ImmutableList<String> DEFAULT_PACKAGE_PATH = ImmutableList.of("%workspace%");

  // Native Java deps are all linked into a single file, which is named with this value + ".so".
  public static final String NATIVE_DEPS_LIB_SUFFIX = "_nativedeps";

  // Locations of implicit Android SDK dependencies.
  public static final String ANDROID_DEFAULT_SDK = "//external:android/sdk".toString();

  // If the --fat_apk_cpu flag is not set, we use this as the default value.
  public static final ImmutableList<String> ANDROID_DEFAULT_FAT_APK_CPUS =
      ImmutableList.<String>of("armeabi-v7a");

  // Most other tools dependencies use this; we plan to split it into per-language repositories.
  public static final String TOOLS_REPOSITORY = "@bazel_tools".toString();
}
