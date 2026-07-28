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
 * {@link LocalJobserver.Backend}. The semaphore's max count is fixed at creation to the local CPU
 * budget, which both caps refills and bounds the drain, so it relies on no undocumented APIs.
 */
public final class WindowsJobserverBackend extends LocalJobserver.Backend {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final int maxTokens;
  private long handle;
  private boolean open;

  /**
   * @param maxTokens the semaphore's fixed max count and thus the largest the token pool may grow
   *     to; size it to the local CPU budget so an over-provisioned budget is honored instead of
   *     capped at the core count.
   */
  public WindowsJobserverBackend(int maxTokens) {
    this.maxTokens = maxTokens;
  }

  @Override
  public String start() throws IOException {
    String name = "bazel-jobserver-" + UUID.randomUUID();
    Long h = WindowsSemaphore.createSemaphore(name, maxTokens);
    if (h == null) {
      throw new IOException("CreateSemaphore failed for " + name);
    }
    this.handle = h;
    this.open = true;
    return name;
  }

  // Windows has no non-mutating semaphore-count query, so peek by acquiring every idle token until
  // tryAcquire fails. The superclass derives held = max(0, issued - available), so even a tool that
  // over-releases (available > issued) is absorbed rather than corrupting the count.
  @Override
  protected int drainPool() throws IOException {
    int available = 0;
    while (WindowsSemaphore.tryAcquire(handle)) {
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
