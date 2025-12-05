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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.IgnoredSubdirectories;
import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsProvider;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.CountDownLatch;

/**
 * A {@link DiffAwareness} that use fsevents to watch the filesystem to use in lieu of {@link
 * LocalDiffAwareness}.
 *
 * <p>On OS X, the local diff awareness cannot work because WatchService is dummy and do polling,
 * which is slow (https://bugs.openjdk.java.net/browse/JDK-7133447).
 */
public final class MacOSXFsEventsDiffAwareness extends LocalDiffAwareness {
  private final double latency;
  private final IgnoredSubdirectories ignoredPaths;
  private final FsEventsNativeDepsService service;

  private boolean closed;
  private boolean opened;

  /**
   * Watch changes on the file system under <code>watchRoot</code> with a granularity of <code>delay
   * </code> seconds.
   */
  MacOSXFsEventsDiffAwareness(
      Path watchRoot,
      IgnoredSubdirectories ignoredPaths,
      double latency,
      FsEventsNativeDepsService fsEventsNativeDepsService) {
    super(watchRoot);
    this.ignoredPaths = ignoredPaths;
    this.latency = latency;
    this.service = fsEventsNativeDepsService;
  }

  /** Watch changes on the file system under <code>watchRoot</code> with a granularity of 5ms. */
  MacOSXFsEventsDiffAwareness(
      Path watchRoot,
      IgnoredSubdirectories ignoredPaths,
      FsEventsNativeDepsService fsEventsNativeDepsService) {
    this(watchRoot, ignoredPaths, 0.005, fsEventsNativeDepsService);
  }

  /**
   * Helper function to start the watch of <code>paths</code>, which is expected to be an array of
   * byte arrays containing the UTF-8 bytes of the paths to watch, called by the constructor.
   */
  private void create(byte[][] paths, byte[][] excludedPaths, double latency) {
    service.createFsEvents(paths, excludedPaths, latency);
  }

  /**
   * Runs the main loop to listen for fsevents.
   *
   * @param listening latch that is decremented when the fsevents queue has been set up. The caller
   *     must wait until this happens before polling for events to ensure no events are lost between
   *     when this function returns and when the queue is listening.
   */
  private void run(CountDownLatch listening) {
    service.runFsEvents(listening);
  }

  private void init() {
    // The code below is based on the assumption that init() can never fail, which is currently the
    // case; if you change init(), then you also need to update {@link #getCurrentView}.
    // TODO(jmmv): This can break if the user interrupts as anywhere in this function.
    Preconditions.checkState(!opened);
    opened = true;
    // TODO: Also cover otherwise literal patterns of the form dir/**.
    var excludedPaths =
        ignoredPaths.prefixes().stream()
            // FSEvents only supports up to 8 excluded paths.
            .limit(8)
            .map(PathFragment::getPathString)
            // The prefixes are all absolute paths converted to relative paths via
            // PathFragment#toRelative.
            .map(path -> "/" + path)
            .map(StringEncoding::internalToUnicode)
            .map(path -> path.getBytes(UTF_8))
            .toArray(byte[][]::new);
    create(
        new byte[][] {watchRoot.toAbsolutePath().toString().getBytes(UTF_8)},
        excludedPaths,
        latency);

    // Start a thread that just contains the OS X run loop.
    CountDownLatch listening = new CountDownLatch(1);
    new Thread(() -> MacOSXFsEventsDiffAwareness.this.run(listening), "osx-fs-events").start();
    try {
      listening.await();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  /** Close this watch service, this service should not be used any longer after closing. */
  @Override
  public void close() {
    if (opened) {
      Preconditions.checkState(!closed);
      closed = true;
      doClose();
    }
  }

  /** JNI code stopping the main loop and shutting down listening to FSEvents. */
  private void doClose() {
    service.doCloseFsEvents();
  }

  /**
   * JNI code returning the list of absolute path modified since last call.
   *
   * @return the array of paths (in the form of byte arrays containing the UTF-8 representation)
   *     modified since the last call, or null if we can't precisely tell what changed
   */
  private byte[][] poll() {
    return service.pollFsEvents();
  }

  @Override
  public View getCurrentView(OptionsProvider options) throws BrokenDiffAwarenessException {
    if (!service.isJniAvailable()) {
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
    byte[][] polledPaths = poll();
    if (polledPaths == null) {
      return EVERYTHING_MODIFIED;
    } else {
      ImmutableSet.Builder<Path> paths = ImmutableSet.builder();
      for (byte[] pathBytes : polledPaths) {
        paths.add(Paths.get(new String(pathBytes, UTF_8)));
      }
      return newView(paths.build());
    }
  }
}
