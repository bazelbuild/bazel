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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r12;

import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.ApiLevel;
import com.google.devtools.build.lib.events.EventHandler;

/** Class which encodes information from the Android NDK makefiles about API levels. */
public class ApiLevelR12 extends ApiLevel {

  private static final ImmutableListMultimap<String, String> API_LEVEL_TO_ARCHITECTURES =
      ImmutableListMultimap.<String, String>builder()
          .putAll("8", "arm")
          .putAll("9", "arm", "mips", "x86")
          .putAll("12", "arm", "mips", "x86")
          .putAll("13", "arm", "mips", "x86")
          .putAll("14", "arm", "mips", "x86")
          .putAll("15", "arm", "mips", "x86")
          .putAll("16", "arm", "mips", "x86")
          .putAll("17", "arm", "mips", "x86")
          .putAll("18", "arm", "mips", "x86")
          .putAll("19", "arm", "mips", "x86")
          .putAll("21", "arm", "mips", "x86", "arm64", "mips64", "x86_64")
          .putAll("23", "arm", "mips", "x86", "arm64", "mips64", "x86_64")
          .putAll("24", "arm", "mips", "x86", "arm64", "mips64", "x86_64")
          .build();

  /**
   * Per the "special cases" listed in build/core/add-application.mk:
   *
   * <pre>
   *   SPECIAL CASES:
   *   1) android-6 and android-7 are the same thing as android-5
   *   2) android-10 and android-11 are the same thing as android-9
   *   3) android-20 is the same thing as android-19
   *   4) android-21 and up are the same thing as android-21
   *   ADDITIONAL CASES for remote server where total number of files is limited
   *   5) android-13 is the same thing as android-12
   *   6) android-15 is the same thing as android-14
   *   7) android-17 is the same thing as android-16
   * </pre>
   */
  private static final ImmutableMap<String, String> API_EQUIVALENCIES =
      ImmutableMap.<String, String>builder()
          // Case 1
          .put("6", "5")
          .put("7", "5")
          .put("8", "8")

          // Case 2
          .put("9", "9")
          .put("10", "9")
          .put("11", "9")

          // Case 5
          .put("12", "12")
          .put("13", "12")

          // Case 6
          .put("14", "14")
          .put("15", "14")

          // Case 7
          .put("16", "16")
          .put("17", "16")
          .put("18", "18")

          // Case 3
          .put("19", "19")
          .put("20", "19")

          // Case 4
          .put("21", "21")
          .put("22", "21")
          .put("23", "23")
          .put("24", "24")
          .build();

  public ApiLevelR12(EventHandler eventHandler, String repositoryName, String apiLevel) {
    super(API_LEVEL_TO_ARCHITECTURES, API_EQUIVALENCIES, eventHandler, repositoryName, apiLevel);
  }
}
