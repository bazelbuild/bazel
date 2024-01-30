// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import com.google.common.flogger.GoogleLogger;
import java.lang.management.ManagementFactory;
import javax.management.ObjectName;

/**
 * A helper class to offset some items from the final heap metrics.
 *
 * <p>Context: b/311665999.
 */
public final class HeapOffsetHelper {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private HeapOffsetHelper() {}

  /**
   * Workaround for the FillerArray issue with JDK21. TODO(b/311665999) Remove this ASAP.
   *
   * <p>With JDK21, an arbitrary amount of FillerArray instances is retained on the heap at the end
   * of each command, which fluctuates our heap metrics a lot. This method looks into the heap
   * histogram and gets the total #bytes attributed to the FillerArray. We'll offset that against
   * the final value to make it more stable.
   */
  @SuppressWarnings("StringSplitter")
  public static long getSizeOfFillerArrayOnHeap() {
    long sizeInBytes = 0;
    try {
      String histogram =
          (String)
              ManagementFactory.getPlatformMBeanServer()
                  .invoke(
                      new ObjectName("com.sun.management:type=DiagnosticCommand"),
                      "gcClassHistogram",
                      new Object[] {null},
                      new String[] {"[Ljava.lang.String;"});
      for (String line : histogram.split("\n")) {
        // ["", <num>, <#instances>, <#bytes>, <class name>]
        if (line.contains("jdk.internal.vm.FillerArray")) {
          sizeInBytes += Long.parseLong(line.split("\\s+")[3]);
        }
      }
    } catch (Exception e) {
      // Swallow all exceptions.
      logger.atWarning().withCause(e).log(
          "Failed to obtain the size of jdk.internal.vm.FillerArray");
    }
    if (sizeInBytes > 0) {
      logger.atInfo().log(
          "Offsetting %d bytes of jdk.internal.vm.FillerArray in the retained heap metric.",
          sizeInBytes);
    }
    return sizeInBytes;
  }
}
