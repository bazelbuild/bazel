// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools;

import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;

/**
 * Class which encodes information from the Android NDK makefiles about API levels. 
 */
class ApiLevel {

  /**
   * Maps an Android API level to the architectures that that level supports.
   * Based on the directories in the "platforms" directory in the NDK.
   */
  private static final ImmutableListMultimap<String, String> API_LEVEL_TO_ARCHITECTURES =
      ImmutableListMultimap.<String, String>builder()
          .putAll("3", "arm")
          .putAll("4", "arm")
          .putAll("5", "arm")
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
          .build();

  /**
   * Maps an API level to it's equivalent API level in the NDK.
   *
   * <p>Per the "special cases" listed in build/core/add-application.mk:
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
          .put("3", "3")
          .put("4", "4")
    
          // Case 1
          .put("5", "5")
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
    
          .build();

  private final String correctedApiLevel;

  ApiLevel(EventHandler eventHandler, String repositoryName, String apiLevel) {
    this.correctedApiLevel = getCorrectedApiLevel(eventHandler, repositoryName, apiLevel);
  }

  /**
   * Translates the given API level to the equivalent API level in the NDK.
   */
  static String getCorrectedApiLevel(
      EventHandler eventHandler, String repositoryName, String apiLevel) {

    String correctedApiLevel = API_EQUIVALENCIES.get(apiLevel);
    if (correctedApiLevel == null) {
      // The user specified an API level we don't know about. Default to the most recent API level.
      // This relies on the entries being added in sorted order.
      String latestApiLevel = Iterables.getLast(API_EQUIVALENCIES.keySet());
      correctedApiLevel = API_EQUIVALENCIES.get(latestApiLevel);

      eventHandler.handle(Event.warn(String.format(
          "API level %s specified by android_ndk_repository '%s' is not available. "
          + "Using latest known API level %s",
          apiLevel, repositoryName, latestApiLevel)));
    }
    return correctedApiLevel;
  }

  String getCpuCorrectedApiLevel(String targetCpu) {

    // Check that this API level supports the given cpu architecture (eg 64 bit is supported on only
    // 21+).
    if (!API_LEVEL_TO_ARCHITECTURES.containsEntry(correctedApiLevel, targetCpu)) {
      // If the given API level does not support the given architecture, find an API level that
      // does support this architecture. A warning isn't printed because the crosstools for
      // architectures that aren't supported by this API level are generated anyway, even if the
      // user doesn't intend to use them (eg, if they're building for only 32 bit archs, the
      // crosstools for the 64 bit toolchains are generated regardless).
      // API_LEVEL_TO_ARCHITECTURES.inverse() returns a map of architectures to the APIs that
      // support that architecture.
      return Iterables.getLast(API_LEVEL_TO_ARCHITECTURES.inverse().get(targetCpu));
    }

    return correctedApiLevel;
  }

}

