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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r10e.NdkMajorRevisionR10;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r11.NdkMajorRevisionR11;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r12.NdkMajorRevisionR12;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r13.NdkMajorRevisionR13;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;

/**
 * Helper methods for generating a CrosstoolRelease proto for the Android NDK based on a particular
 * NDK release.
 */
public final class AndroidNdkCrosstools {
  private AndroidNdkCrosstools() {}

  // NDK minor revisions should be backwards compatible within a major revision, so all that needs
  // to be tracked here are the major revision numbers.
  public static final ImmutableMap<String, NdkMajorRevision> KNOWN_NDK_MAJOR_REVISIONS =
      ImmutableMap.of(
          "10", new NdkMajorRevisionR10(),
          "11", new NdkMajorRevisionR11(),
          "12", new NdkMajorRevisionR12(),
          "13", new NdkMajorRevisionR13());
  public static final String LATEST_KNOWN_REVISION =
      Iterables.getLast(KNOWN_NDK_MAJOR_REVISIONS.keySet());

  /**
   * Exception thrown when there is an error creating the crosstools file.
   */
  public static class NdkCrosstoolsException extends Exception {
    private NdkCrosstoolsException(String msg) {
      super(msg);
    }
  }

  public static CrosstoolRelease create(
      NdkRelease ndkRelease,
      NdkPaths ndkPaths,
      StlImpl stlImpl,
      String hostPlatform) {
    NdkMajorRevision ndkMajorRevision;
    if (ndkRelease.isValid) {
      // NDK minor revisions should be backwards compatible within a major revision, so it should be
      // enough to check the major revision of the release.
      ndkMajorRevision = KNOWN_NDK_MAJOR_REVISIONS.get(ndkRelease.majorRevision);
    } else {
      // If the NDK revision isn't valid, try using the latest one we know about.
      ndkMajorRevision = KNOWN_NDK_MAJOR_REVISIONS.get(LATEST_KNOWN_REVISION);
    }
    return ndkMajorRevision.crosstoolRelease(ndkPaths, stlImpl, hostPlatform);
  }

  public static String getHostPlatform(NdkRelease ndkRelease) throws NdkCrosstoolsException {
    String hostOs;
    switch (OS.getCurrent()) {
      case DARWIN:
        hostOs = "darwin";
        break;
      case LINUX:
        hostOs = "linux";
        break;
      case WINDOWS:
        hostOs = "windows";
        if (!ndkRelease.is64Bit) {
          // 32-bit windows paths don't have the "-x86" suffix in the NDK (added below), but
          // 64-bit windows does have the "-x86_64" suffix.
          return hostOs;
        }
        break;
      case UNKNOWN:
      default:
        throw new NdkCrosstoolsException(
            String.format("NDK does not support the host platform \"%s\"", OS.getCurrent()));
    }

    // Use the arch from the NDK rather than detecting the actual platform, since it's possible
    // to use the 32-bit NDK on a 64-bit machine.
    String hostArch = ndkRelease.is64Bit ? "x86_64" : "x86";

    return hostOs + "-" + hostArch;
  }

  public static boolean isKnownNDKRevision(NdkRelease ndkRelease) {
    return KNOWN_NDK_MAJOR_REVISIONS.containsKey(ndkRelease.majorRevision);
  } 
}