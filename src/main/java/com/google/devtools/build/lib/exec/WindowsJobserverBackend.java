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
import javax.annotation.Nullable;

/**
 * The Windows {@link LocalJobserver.Backend}: a named semaphore.
 *
 * <p>Forward-only pool sizing: grow toward the idle-CPU target with {@code ReleaseSemaphore} and
 * shrink with best-effort, non-blocking {@code WaitForSingleObject(handle, 0)} polls, since Windows
 * has no documented way to read a semaphore's count without mutating it. A failed shrink means a
 * tool currently holds that token, so the number of failures on a shrink tick is the held-token
 * estimate. Unlike the fifo backend's exact per-tick accounting, this is only refreshed while
 * shrinking and so is coarser; it is left unchanged on grow/steady ticks rather than guessed at.
 * That is why it is recommended to pair this feature with {@code
 * --experimental_cpu_load_scheduling} on Windows, under which held tokens are not consulted for
 *  admission at all (see {@code ResourceManager#isCpuAvailable}) — token holders' threads already
 *  show up in the measured system load.
 */
public final class WindowsJobserverBackend implements LocalJobserver.Backend {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final String dirPath;

  private long handle;
  private boolean open;
  private int maxTokens;
  private long granted = 0;
  private long reclaimed = 0;
  private int lastHeld = 0;

  public WindowsJobserverBackend(String dirPath) {
    this.dirPath = dirPath;
  }

  @Override
  public String start() throws IOException {
    String name = "bazel-jobserver-" + Integer.toHexString(dirPath.hashCode());
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
  public int tick(int targetTokens) {
    long outstanding = granted - reclaimed;
    long delta = targetTokens - outstanding;
    int held = lastHeld;
    if (delta > 0) {
      int grow = (int) Math.min(delta, maxTokens - outstanding);
      if (grow > 0 && WindowsSemaphore.release(handle, grow)) {
        granted += grow;
      }
    } else if (delta < 0) {
      int toReclaim = (int) -delta;
      int failed = 0;
      for (int i = 0; i < toReclaim; i++) {
        if (WindowsSemaphore.tryAcquire(handle)) {
          reclaimed++;
        } else {
          failed++;
        }
      }
      held = failed;
    }
    lastHeld = held;
    return held;
  }

  @Override
  public void wakeForShutdown() {
    // The semaphore backend never blocks in tick(), so its poll loop wakes on its own.
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
