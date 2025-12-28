// Copyright 2025 The Bazel Authors. All rights reserved.
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
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.nio.channels.FileChannel;
import java.util.concurrent.atomic.AtomicBoolean;

/** Utility methods for {@link FileChannel}. */
public final class FileChannels {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final AtomicBoolean alreadyLogged = new AtomicBoolean(false);

  private static final MethodHandle SET_UNINTERRUPTIBLE;

  static {
    MethodHandle handle = null;
    try {
      handle =
          MethodHandles.lookup()
              .unreflect(
                  Class.forName("sun.nio.ch.FileChannelImpl")
                      .getDeclaredMethod("setUninterruptible"));
    } catch (ReflectiveOperationException e) {
      // Ignore: maybe we're using a JDK that doesn't provide this API.
      logger.atWarning().withCause(e).log(
          "Failed to obtain method handle for FileChannelImpl.setUninterruptible");
    } finally {
      SET_UNINTERRUPTIBLE = handle;
    }
  }

  /**
   * Makes the given channel uninterruptible.
   *
   * <p>This uses an internal OpenJDK API and may silently fail if it's not available.
   */
  public static void setUninterruptible(FileChannel channel) {
    if (SET_UNINTERRUPTIBLE != null) {
      try {
        SET_UNINTERRUPTIBLE.invoke(channel);
      } catch (Throwable e) {
        // Ignore: maybe we're using a JDK that doesn't provide this API.
        if (alreadyLogged.compareAndSet(false, true)) {
          logger.atWarning().withCause(e).log(
              "Failed to call FileChannelImpl.setUninterruptible (only logged once)");
        }
      }
    }
  }

  private FileChannels() {}
}
