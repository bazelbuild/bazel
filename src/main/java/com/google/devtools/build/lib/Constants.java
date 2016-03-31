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

/**
 * A temporary class of constants; these encode differences between Google's internal setup and
 * Bazel. We're working to remove this class, which requires cleaning up our internal code base.
 * Please don't add anything here unless you know what you're doing.
 */
public final class Constants {
  private Constants() {}

  // Google's internal name for Bazel is 'Blaze', and it will take some more time to change that.
  public static final String PRODUCT_NAME = "bazel";

  // Native Java deps are all linked into a single file, which is named with this value + ".so".
  public static final String NATIVE_DEPS_LIB_SUFFIX = "_nativedeps";

  // Locations of implicit Android SDK dependencies.
  public static final String ANDROID_DEFAULT_SDK = "//external:android/sdk";

  // Most other tools dependencies use this; we plan to split it into per-language repositories.
  public static final String TOOLS_REPOSITORY = "@bazel_tools";
}
