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
package com.google.devtools.build.lib.testutil;

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;

/**
 * Test utilities that help waiting for the filesystem timestamp granularity.
 *
 * <p>It is necessary to wait for the timestamp granularity if a test asserts changes in the mtime
 * or ctime of a file.
 */
public abstract class TimestampGranularityUtils {

  private TimestampGranularityUtils() {}

  /** Wait enough such that changes to a file with the given ctime will have observable effects. */
  public static void waitForTimestampGranularity(long ctimeMillis, OutErr outErr) {
    TimestampGranularityMonitor tsgm = new TimestampGranularityMonitor(BlazeClock.instance());
    tsgm.setCommandStartTime();
    tsgm.notifyDependenceOnFileTime(null, ctimeMillis);
    tsgm.waitForTimestampGranularity(outErr);
  }
}
