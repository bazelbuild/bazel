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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.runtime.BlazeService;
import java.util.concurrent.CountDownLatch;

/** Service interface for Skyframe native dependencies. */
public interface FsEventsNativeDepsService extends BlazeService {
  /** Returns true if the JNI code is available. */
  boolean isJniAvailable();

  /**
   * Helper function to start the watch of <code>paths</code>, which is expected to be an array of
   * byte arrays containing the UTF-8 bytes of the paths to watch, called by the constructor.
   *
   * @param paths the paths to watch.
   * @param excludedPaths the paths to exclude.
   * @param latency the latency to use for the FSEvents stream.
   */
  void createFsEvents(byte[][] paths, byte[][] excludedPaths, double latency);

  /**
   * Runs the main loop to listen for fsevents.
   *
   * @param listening latch that is decremented when the fsevents queue has been set up. The caller
   *     must wait until this happens before polling for events to ensure no events are lost between
   *     when this function returns and when the queue is listening.
   */
  void runFsEvents(CountDownLatch listening);

  /** JNI code stopping the main loop and shutting down listening to FSEvents. */
  void doCloseFsEvents();

  /**
   * JNI code returning the list of absolute path modified since last call.
   *
   * @return the array of paths (in the form of byte arrays containing the UTF-8 representation)
   *     modified since the last call, or null if we can't precisely tell what changed
   */
  byte[][] pollFsEvents();
}
