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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools;

import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r10e.ApiLevelR10e;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r11.ApiLevelR11;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r12.ApiLevelR12;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;

/**
 * Base class for ApiLevels, which provides a factory function to create an ApiLevel based on an
 * NDK release.
 */
public abstract class ApiLevel {

  public static ApiLevel getApiLevel(
      NdkRelease release, EventHandler eventHandler, String repositoryName, String apiLevel) {
    if ("10".equals(release.majorRevision)) {
      return new ApiLevelR10e(eventHandler, repositoryName, apiLevel);
    } else if ("11".equals(release.majorRevision)) {
      return new ApiLevelR11(eventHandler, repositoryName, apiLevel);
    } else {
      return new ApiLevelR12(eventHandler, repositoryName, apiLevel);
    }
  }
  
  /**
   * Maps an Android API level to the architectures that that level supports.
   * Based on the directories in the "platforms" directory in the NDK.
   */
  private final ImmutableListMultimap<String, String> apiLevelToArchitectures;

  /**
   * Maps an API level to its equivalent API level in the NDK.
   */
  private final ImmutableMap<String, String> apiEquivalencies;

  private final String correctedApiLevel;
  
  protected ApiLevel(
      ImmutableListMultimap<String, String> apiLevelToArchitectures,
      ImmutableMap<String, String> apiEquivalencies,
      EventHandler eventHandler,
      String repositoryName,
      String apiLevel) {

    this.apiLevelToArchitectures = apiLevelToArchitectures;
    this.apiEquivalencies = apiEquivalencies;
    this.correctedApiLevel = getCorrectedApiLevel(eventHandler, repositoryName, apiLevel);
  }

  /**
   * Translates the given API level to the equivalent API level in the NDK.
   */
  private String getCorrectedApiLevel(
      EventHandler eventHandler, String repositoryName, String apiLevel) {

    String correctedApiLevel = apiEquivalencies.get(apiLevel);
    if (correctedApiLevel == null) {
      // The user specified an API level we don't know about. Default to the most recent API level.
      // This relies on the entries being added in sorted order.
      String latestApiLevel = Iterables.getLast(apiEquivalencies.keySet());
      correctedApiLevel = apiEquivalencies.get(latestApiLevel);

      eventHandler.handle(Event.warn(String.format(
          "API level %s specified by android_ndk_repository '%s' is not available. "
          + "Using latest known API level %s",
          apiLevel, repositoryName, latestApiLevel)));
    }
    return correctedApiLevel;
  }

  public String getCpuCorrectedApiLevel(String targetCpu) {

    // Check that this API level supports the given cpu architecture (eg 64 bit is supported on only
    // 21+).
    if (!apiLevelToArchitectures.containsEntry(correctedApiLevel, targetCpu)) {
      // If the given API level does not support the given architecture, find an API level that
      // does support this architecture. A warning isn't printed because the crosstools for
      // architectures that aren't supported by this API level are generated anyway, even if the
      // user doesn't intend to use them (eg, if they're building for only 32 bit archs, the
      // crosstools for the 64 bit toolchains are generated regardless).
      // API_LEVEL_TO_ARCHITECTURES.inverse() returns a map of architectures to the APIs that
      // support that architecture.
      return Iterables.getLast(apiLevelToArchitectures.inverse().get(targetCpu));
    }

    return correctedApiLevel;
  }
}

