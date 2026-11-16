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
package com.google.devtools.build.lib.clock;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import com.google.devtools.build.lib.util.CPU;
import com.google.devtools.build.lib.util.OS;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.invoke.MethodHandle;
import javax.annotation.Nullable;

/**
 * Class provides a simple clock implementation used by the tool. By default it uses {@link System}
 * class.
 */
public class JavaClock implements Clock {

  // On macOS, System.nanoTime() reads mach_absolute_time() and then updates a global variable with
  // a CAS to ensure monotonicity, which makes it slow when called from many threads. See
  // https://github.com/openjdk/jdk/blob/ad645ad30bf45420174153859d87a6110412b115/src/hotspot/os/bsd/os_bsd.cpp#L841-L864.
  // On Apple silicon, the clock is a counter shared by all cores and thus already monotonic, so
  // read it directly through clock_gettime_nsec_np(), which Apple recommends over
  // mach_absolute_time() and which converts to nanoseconds the same way. See
  // https://developer.apple.com/documentation/kernel/1462446-mach_absolute_time and
  // https://github.com/apple-oss-distributions/Libc/blob/4e34d0559e3a1b081afeb8604d9e204a1f31321d/gen/clock_gettime.c#L121-L135.
  // On Intel Macs, it reads the TSC of the current core, which isn't guaranteed to be synchronized
  // with those of other cores. See
  // https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/i386/mp.c#L412-L421.
  @Nullable private static final MethodHandle CLOCK_GETTIME_NSEC_NP;

  // https://github.com/apple-oss-distributions/Libc/blob/4e34d0559e3a1b081afeb8604d9e204a1f31321d/include/_time.h#L220
  private static final int CLOCK_UPTIME_RAW = 8;

  static {
    MethodHandle clockGettimeNsecNp = null;
    if (OS.getCurrent() == OS.DARWIN
        && CPU.getCurrent() == CPU.AARCH64
        && JavaClock.class.getModule().isNativeAccessEnabled()) {
      Linker linker = Linker.nativeLinker();
      // __uint64_t clock_gettime_nsec_np(clockid_t __clock_id), where clockid_t is an enum. See
      // https://github.com/apple-oss-distributions/Libc/blob/4e34d0559e3a1b081afeb8604d9e204a1f31321d/include/_time.h#L241
      clockGettimeNsecNp =
          linker
              .defaultLookup()
              .find("clock_gettime_nsec_np")
              .map(
                  symbol ->
                      linker.downcallHandle(
                          symbol,
                          FunctionDescriptor.of(JAVA_LONG, JAVA_INT),
                          Linker.Option.critical(/* allowHeapAccess= */ false)))
              .orElse(null);
    }
    CLOCK_GETTIME_NSEC_NP = clockGettimeNsecNp;
  }

  public JavaClock() {
  }

  @Override
  public long currentTimeMillis() {
    return System.currentTimeMillis();
  }

  @Override
  public long nanoTime() {
    if (CLOCK_GETTIME_NSEC_NP != null) {
      try {
        return (long) CLOCK_GETTIME_NSEC_NP.invokeExact(CLOCK_UPTIME_RAW);
      } catch (Throwable t) {
        throw new AssertionError(t);
      }
    }
    // Note that some JVM implementations of System#nanoTime don't yield a non-decreasing
    // sequence of values.
    return System.nanoTime();
  }
}
