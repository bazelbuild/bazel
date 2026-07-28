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

package com.google.devtools.build.lib.windows;

import com.google.devtools.build.lib.jni.JniLoader;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Named-semaphore primitive backing {@code LocalJobserver} on Windows.
 */
public final class WindowsSemaphore {

  static {
    JniLoader.loadJni();
  }

  private WindowsSemaphore() {}

  /**
   * Creates a named semaphore via {@code CreateSemaphoreW} with initial count 0 and maximum count
   * {@code maxCount}, returning its OS handle, or {@code null} on failure.
   */
  @Nullable
  public static Long createSemaphore(String name, int maxCount) {
    long handle = createSemaphore0(name, 0, maxCount);
    return handle == 0 ? null : handle;
  }

  private static native long createSemaphore0(String name, int initialCount, int maxCount);

  /**
   * Adds {@code delta} to the semaphore's count via {@code ReleaseSemaphore} (i.e. grows the pool).
   * A non-positive {@code delta} is a no-op that trivially succeeds, since {@code ReleaseSemaphore}
   * itself requires a positive count. Returns false on failure (e.g. the count would exceed the
   * maximum set at creation).
   */
  public static boolean release(long handle, int delta) {
    return delta <= 0 || release0(handle, delta);
  }

  private static native boolean release0(long handle, int delta);

  /**
   * Attempts to take one token with a zero-timeout {@code WaitForSingleObject}. Returns true if a
   * token was acquired (and thus removed from the semaphore's count), false if none was available,
   * and throws if the wait itself failed.
   */
  public static boolean tryAcquire(long handle) throws IOException {
    return switch (tryAcquire0(handle)) {
      case 1 -> true;
      case 0 -> false;
      default -> throw new IOException("WaitForSingleObject failed");
    };
  }

  private static native int tryAcquire0(long handle);

  /** Closes the semaphore handle via {@code CloseHandle}. */
  public static native void close(long handle);
}
