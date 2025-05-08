// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.metrics;

import com.sun.management.GarbageCollectionNotificationInfo;

/** Utility methods for garbage collection metrics. */
public final class GarbageCollectionMetricsUtils {
  private GarbageCollectionMetricsUtils() {}

  public static boolean isTenuredSpace(String name) {
    return "CMS Old Gen".equals(name)
        || "G1 Old Gen".equals(name)
        || "PS Old Gen".equals(name)
        || "Tenured Gen".equals(name)
        || "Shenandoah".equals(name)
        || "ZHeap".equals(name);
  }

  public static boolean isFullGc(GarbageCollectionNotificationInfo info) {
    return info.getGcAction().equals("end of major GC");
  }
}
