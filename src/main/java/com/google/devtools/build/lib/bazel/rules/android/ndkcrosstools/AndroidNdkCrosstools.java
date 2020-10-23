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
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r15.NdkMajorRevisionR15;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r17.NdkMajorRevisionR17;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r18.NdkMajorRevisionR18;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r19.NdkMajorRevisionR19;
import com.google.devtools.build.lib.util.OS;
import java.util.Map;

/**
 * Helper methods for generating a CrosstoolRelease proto for the Android NDK based on a particular
 * NDK release.
 */
public final class AndroidNdkCrosstools {
  private AndroidNdkCrosstools() {}

  // NDK minor revisions should be backwards compatible within a major revision, so all that needs
  // to be tracked here are the major revision numbers.
  public static final ImmutableMap<Integer, NdkMajorRevision> KNOWN_NDK_MAJOR_REVISIONS =
      new ImmutableMap.Builder<Integer, NdkMajorRevision>()
          .put(10, new NdkMajorRevisionR10())
          .put(11, new NdkMajorRevisionR11())
          .put(12, new NdkMajorRevisionR12())
          .put(13, new NdkMajorRevisionR13("3.8.256229"))
          // The only difference between the NDK13 and NDK14 CROSSTOOLs is the version of clang in
          // built-in includes paths, so we can reuse everything else.
          .put(14, new NdkMajorRevisionR13("3.8.275480"))
          .put(15, new NdkMajorRevisionR15("5.0.300080"))
          // The only difference between r15 and r16 is that old headers are removed, forcing
          // usage of the unified headers. Support for unified headers were added in r15.
          .put(16, new NdkMajorRevisionR15("5.0.300080")) // no changes relevant to Bazel
          .put(17, new NdkMajorRevisionR17("6.0.2"))
          .put(18, new NdkMajorRevisionR18("7.0.2"))
          .put(19, new NdkMajorRevisionR19("8.0.2"))
          .put(20, new NdkMajorRevisionR19("8.0.7")) // no changes relevant to Bazel
          .put(21, new NdkMajorRevisionR19("9.0.8")) // no changes relevant to Bazel
          .build();

  public static final Map.Entry<Integer, NdkMajorRevision> LATEST_KNOWN_REVISION =
      Iterables.getLast(KNOWN_NDK_MAJOR_REVISIONS.entrySet());

  /**
   * Exception thrown when there is an error creating the crosstools file.
   */
  public static class NdkCrosstoolsException extends Exception {
    private NdkCrosstoolsException(String msg) {
      super(msg);
    }
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
