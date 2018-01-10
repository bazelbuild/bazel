// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

/**
 * DO NOT USE in Bazel.
 *
 * <p>A temporary, and very regrettable Google-only hack that allows us to support targeting other
 * platforms in certain cases.
 */
public class FakeCPU {

  private FakeCPU() {
    // Private constructor to prohibit creating objects.
  }

  /**
   * These are fake CPU values used to indicate that amd64 OSX / Windows should be
   * the targeted architecture and platform.  This is a hack to support compiling
   * Go targeting OSX and Windows, until we have proper support for this kind of thing.
   * It is largely unsupported.
   */
  public static final String DARWIN_FAKE_CPU = "x86_64-darwin";
  public static final String WINDOWS_FAKE_CPU_64 = "x86_64-windows";
  public static final String WINDOWS_FAKE_CPU_32 = "x86_32-windows";

  public static boolean isFakeCPU(String cpu) {
    return DARWIN_FAKE_CPU.equals(cpu) || WINDOWS_FAKE_CPU_32.equals(cpu)
        || WINDOWS_FAKE_CPU_64.equals(cpu);
  }

  /**
   * Returns the real CPU for a (possible) fake CPU. If isFakeCPU(fakeCPU)
   * returns true,
   * this method will return the actual target CPU that should be used.
   * Otherwise, it
   * will simply return fakeCPU.
   */
  public static String getRealCPU(String fakeCPU) {
    if (isFakeCPU(fakeCPU)) {
      // We have a special fake CPU for 32 bit Windows binaries.
      if (WINDOWS_FAKE_CPU_32.equals(fakeCPU)) {
        return "piii";
      }
      // If targeting darwin or windows 64, pretend to be k8 so that we don't need to
      // mess with crosstool configurations. A big fat warning was printed by the
      // ConfigurationFactory warning people that they shouldn't expect anything
      // other than go_{binary,library} to work, so the spurious
      // k8 results we will return for other languages are fine.
      return "k8";
    } else {
      return fakeCPU;
    }
  }
}
