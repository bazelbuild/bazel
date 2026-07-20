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

/**
 * The Windows {@link LocalJobserver.Backend}: a named semaphore. A client waits on it to take a
 * token and releases it to return one.
 *
 * <p>Peeks the pool by acquiring the available tokens (Windows has no non-mutating count query) and
 * refills by releasing; the shared {@code issued}/{@code available}/{@code held} accounting lives in
 * {@link LocalJobserver.Backend}. Acquiring is bounded by {@code issued} and refills are
 * capped by the semaphore's fixed max count, so it relies on no undocumented APIs.
 */
public final class WindowsJobserverBackend extends LocalJobserver.Backend {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private long handle;
  private boolean open;
  private int maxTokens;

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

  // Windows has no non-mutating semaphore-count query, so peek by acquiring: tryAcquire down to the
  // number issued (each failed acquire means a tool holds that token). The superclass turns the
  // returned available count into held and refills to the new target.
  @Override
  protected int drainPool(int issued) throws IOException {
    int available = 0;
    while (available < issued && WindowsSemaphore.tryAcquire(handle)) {
      available++;
    }
    return available;
  }

  @Override
  protected void refillPool(int count) throws IOException {
    if (count <= 0) {
      return;
    }
    if (!WindowsSemaphore.release(handle, count)) {
      throw new IOException("ReleaseSemaphore failed");
    }
  }

  // The semaphore's max count is fixed at creation; ReleaseSemaphore fails past it.
  @Override
  protected int clampTarget(int targetTokens) {
    return Math.min(targetTokens, maxTokens);
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
