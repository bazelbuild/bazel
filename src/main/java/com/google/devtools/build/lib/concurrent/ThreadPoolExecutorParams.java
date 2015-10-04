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

package com.google.devtools.build.lib.concurrent;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Configuration parameters for {@link ThreadPoolExecutor} construction.
 */
public class ThreadPoolExecutorParams {
  private final int corePoolSize;
  private final int maxPoolSize;
  private final long keepAliveTime;
  private final TimeUnit units;
  private final String poolName;
  private final BlockingQueue<Runnable> workQueue;

  public ThreadPoolExecutorParams(int corePoolSize, int maxPoolSize, long keepAliveTime,
      TimeUnit units, String poolName, BlockingQueue<Runnable> workQueue) {
    this.corePoolSize = corePoolSize;
    this.maxPoolSize = maxPoolSize;
    this.keepAliveTime = keepAliveTime;
    this.units = units;
    this.poolName = poolName;
    this.workQueue = workQueue;
  }

  public int getCorePoolSize() {
    return corePoolSize;
  }

  public int getMaxPoolSize() {
    return maxPoolSize;
  }

  public long getKeepAliveTime() {
    return keepAliveTime;
  }

  public TimeUnit getUnits() {
    return units;
  }

  public String getPoolName() {
    return poolName;
  }

  public BlockingQueue<Runnable> getWorkQueue() {
    return workQueue;
  }
}
