// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.windows.WindowsSemaphore;
import java.io.IOException;
import java.util.UUID;
import javax.annotation.Nullable;

/**
 * The Windows {@link LocalJobserver.Backend}: a named semaphore.
 *
 * <p>Windows has no documented non-mutating semaphore-count query, so each tick non-blockingly
 * drains the available tokens, derives the held count from the distributed total, and releases
 * only the tokens needed for the new target. This briefly withholds idle tokens from clients but
 * gives the same per-tick accounting as the fifo backend without relying on undocumented APIs.
 */
public final class WindowsJobserverBackend implements LocalJobserver.Backend {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private long handle;
  private boolean open;
  private int maxTokens;
  private int outstanding;

  @Override
  public String start() throws IOException {
    String name = "bazel-jobserver-" + UUID.randomUUID();
    int max = Runtime.getRuntime().availableProcessors();
    Long h = WindowsSemaphore.createSemaphore(name, max);
    if (h == null) {
      throw new IOException("CreateSemaphore failed for " + name);
    }
    this.handle = h;
    this.open = true;
    this.maxTokens = max;
    return name;
  }

  @Override
  @Nullable
  public String writableDir() {
    return null; // no filesystem path backs a named semaphore
  }

  @Override
  public int tick(int targetTokens) throws IOException {
    int available = 0;
    while (available < outstanding && WindowsSemaphore.tryAcquire(handle)) {
      available++;
    }
    int held = outstanding - available;
    int target = Math.min(targetTokens, maxTokens);
    int desired = Math.max(0, target - held);
    outstanding = held;
    if (desired > 0) {
      if (!WindowsSemaphore.release(handle, desired)) {
        throw new IOException("ReleaseSemaphore failed");
      }
      outstanding += desired;
    }
    return held;
  }

  @Override
  public void close() {
    if (!open) {
      return;
    }
    WindowsSemaphore.close(handle);
    open = false;
    logger.atInfo().log("Local jobserver semaphore closed");
  }
}
