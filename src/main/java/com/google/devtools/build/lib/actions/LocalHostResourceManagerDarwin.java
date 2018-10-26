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

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.unix.NativePosixSystem;

import java.io.IOException;

/**
 * This class estimates the local host's resource capacity for Darwin.
 */
public class LocalHostResourceManagerDarwin {
  private static final Boolean JNI_UNAVAILABLE =
      "0".equals(System.getProperty("io.bazel.EnableJni"));

  private static int getLogicalCpuCount() throws IOException {
    return (int) NativePosixSystem.sysctlbynameGetLong("hw.logicalcpu");
  }

  private static double getMemoryInMb() throws IOException {
    return NativePosixSystem.sysctlbynameGetLong("hw.memsize") / 1E6;
  }

  public static ResourceSet getLocalHostResources() {
    if (JNI_UNAVAILABLE) {
      return null;
    }
    try {
      int logicalCpuCount = getLogicalCpuCount();
      double ramMb = getMemoryInMb();

      return ResourceSet.create(
          ramMb,
          logicalCpuCount,
          Integer.MAX_VALUE);
    } catch (IOException e) {
      return null;
    }
  }
}
