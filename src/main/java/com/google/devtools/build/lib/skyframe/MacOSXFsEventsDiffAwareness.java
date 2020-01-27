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
import com.google.devtools.common.options.OptionsProvider;
import java.io.File;
import java.nio.file.Path;

/**
 * A {@link DiffAwareness} that use fsevents to watch the filesystem to use in lieu of
 * {@link LocalDiffAwareness}.
 *
 * <p>On OS X, the local diff awareness cannot work because WatchService is dummy and do polling,
 * which is slow (https://bugs.openjdk.java.net/browse/JDK-7133447).
 */
public final class MacOSXFsEventsDiffAwareness extends LocalDiffAwareness {
  private final double latency;

  private boolean closed;

  // Keep a pointer to a native structure in the JNI code (the FsEvents callback needs that
  // structure).
  private long nativePointer;

  private boolean opened;

  /**
   * Watch changes on the file system under <code>watchRoot</code> with a granularity of
   * <code>delay</code> seconds.
   */
  MacOSXFsEventsDiffAwareness(String watchRoot, double latency) {
    super(watchRoot);
    this.latency = latency;
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

  private void init() {
    // The code below is based on the assumption that init() can never fail, which is currently the
    // case; if you change init(), then you also need to update {@link #getCurrentView}.
    Preconditions.checkState(!opened);
    opened = true;
    create(new String[] {watchRootPath.toAbsolutePath().toString()}, latency);
    // Start a thread that just contains the OS X run loop.
    new Thread(() -> MacOSXFsEventsDiffAwareness.this.run(), "osx-fs-events").start();
  }

  /**
   * Close this watch service, this service should not be used any longer after closing.
   */
  @Override
  public void close() {
    if (opened) {
      Preconditions.checkState(!closed);
      closed = true;
      doClose();
    }
  }

  private static final boolean JNI_AVAILABLE;

  /**
   * JNI code stopping the main loop and shutting down listening to FSEvents.
   */
  private native void doClose();

  /**
   * JNI code returning the list of absolute path modified since last call.
   */
  private native String[] poll();

  static {
    boolean loadJniWorked = false;
    try {
      UnixJniLoader.loadJni();
      loadJniWorked = true;
    } catch (UnsatisfiedLinkError ignored) {
      // Unfortunately, we compile this class into the Bazel bootstrap binary, which doesn't have
      // access to the JNI code (to simplify bootstrap). This is the quick and dirty way to
      // hard-disable --watchfs in the bootstrap binary.
    }
    JNI_AVAILABLE = loadJniWorked;
  }

  @Override
  public View getCurrentView(OptionsProvider options)
      throws BrokenDiffAwarenessException {
    if (!JNI_AVAILABLE) {
      return EVERYTHING_MODIFIED;
    }
    // See WatchServiceDiffAwareness#getCurrentView for an explanation of this logic.
    boolean watchFs = options.getOptions(Options.class).watchFS;
    if (watchFs && !opened) {
      init();
    } else if (!watchFs && opened) {
      close();
      throw new BrokenDiffAwarenessException("Switched off --watchfs again");
    } else if (!opened) {
      // The only difference with WatchServiceDiffAwareness#getCurrentView is this if; the init()
      // call above can never fail, so we don't need to re-check the opened flag after init().
      return EVERYTHING_MODIFIED;
    }
    Preconditions.checkState(!closed);
    ImmutableSet.Builder<Path> paths = ImmutableSet.builder();
    for (String path : poll()) {
      paths.add(new File(path).toPath());
    }
    return newView(paths.build());
  }
}
