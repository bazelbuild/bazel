// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.UnixJniLoader;

import java.io.File;
import java.nio.file.Path;

/**
 * A {@link DiffAwareness} that use fsevents to watch the filesystem to use in lieu of
 * {@link LocalDiffAwareness}.
 *
 * <p>
 * On OS X, the local diff awareness cannot work because WatchService is dummy and do polling, which
 * is slow (https://bugs.openjdk.java.net/browse/JDK-7133447).
 */
public final class MacOSXFsEventsDiffAwareness extends LocalDiffAwareness {

  private boolean closed;

  // Keep a pointer to a native structure in the JNI code (the FsEvents callback needs that
  // structure).
  private long nativePointer;

  /**
   * Watch changes on the file system under <code>watchRoot</code> with a granularity of
   * <code>delay</code> seconds.
   */
  MacOSXFsEventsDiffAwareness(String watchRoot, double latency) {
    super(watchRoot);
    create(new String[] {watchRootPath.toAbsolutePath().toString()}, latency);

    // Start a thread that just contains the OS X run loop.
    new Thread(
            new Runnable() {
              @Override
              public void run() {
                MacOSXFsEventsDiffAwareness.this.run();
              }
            })
        .start();
  }

  /**
   * Watch changes on the file system under <code>watchRoot</code> with a granularity of 5ms.
   */
  MacOSXFsEventsDiffAwareness(String watchRoot) {
    this(watchRoot, 0.005);
  }

  /**
   * Helper function to start the watch of <code>paths</code>, called by the constructor.
   */
  private native void create(String[] paths, double latency);

  /**
   * Run the main loop
   */
  private native void run();

  /**
   * Close this watch service, this service should not be used any longer after closing.
   */
  public synchronized void close() {
    Preconditions.checkState(!closed);
    closed = true;
    doClose();
  }

  /**
   * JNI code stopping the main loop and shutting down listening to FSEvents.
   */
  private synchronized native void doClose();

  /**
   * JNI code returning the list of absolute path modified since last call.
   */
  private native String[] poll();

  static {
    UnixJniLoader.loadJni();
  }

  @Override
  public synchronized View getCurrentView() throws BrokenDiffAwarenessException {
    Preconditions.checkState(!closed);
    ImmutableSet.Builder<Path> paths = ImmutableSet.builder();
    for (String path : poll()) {
      paths.add(new File(path).toPath());
    }
    return newView(paths.build());
  }
}
