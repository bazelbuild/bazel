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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r13;

import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.ApiLevel;
import com.google.devtools.build.lib.events.EventHandler;

/** Class which encodes information from the Android NDK makefiles about API levels. */
final class ApiLevelR13 extends ApiLevel {
  /** This is the contents of {@code platforms/android-*} */
  private static final ImmutableListMultimap<String, String> API_LEVEL_TO_ARCHITECTURES =
      ImmutableListMultimap.<String, String>builder()
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

  /** This map fill in the gaps of {@code API_LEVEL_TO_ARCHITECTURES}. */
  private static final ImmutableMap<String, String> API_EQUIVALENCIES =
      ImmutableMap.<String, String>builder()
          .put("8", "8")
          .put("9", "9")
          .put("10", "9")
          .put("11", "9")
          .put("12", "12")
          .put("13", "12")
          .put("14", "14")
          .put("15", "14")
          .put("16", "16")
          .put("17", "16")
          .put("18", "18")
          .put("19", "19")
          .put("20", "19")
          .put("21", "21")
          .put("22", "22")
          .put("23", "23")
          .put("24", "24")
          .put("25", "24")
          .build();

  ApiLevelR13(EventHandler eventHandler, String repositoryName, String apiLevel) {
    super(API_LEVEL_TO_ARCHITECTURES, API_EQUIVALENCIES, eventHandler, repositoryName, apiLevel);
  }
}
