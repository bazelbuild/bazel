// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.common;

import com.google.common.base.MoreObjects;
import com.google.common.base.Stopwatch;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import java.time.Duration;

/** Reentrant wall clock stopwatch and grpc interceptor for network waits. */
@ThreadSafety.ThreadSafe
public class NetworkTime {

  private final Stopwatch wallTime = Stopwatch.createUnstarted();
  private int outstanding = 0;

  public synchronized void start() {
    if (!wallTime.isRunning()) {
      wallTime.start();
    }
    outstanding++;
  }

  public synchronized void stop() {
    if (--outstanding == 0) {
      wallTime.stop();
    }
  }

  public Duration getDuration() {
    return wallTime.elapsed();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("outstanding", outstanding)
        .add("wallTime", wallTime)
        .add("wallTime.isRunning", wallTime.isRunning())
        .toString();
  }

}
