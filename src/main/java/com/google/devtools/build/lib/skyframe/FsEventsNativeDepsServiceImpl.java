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

import com.google.devtools.build.lib.jni.JniLoader;
import java.util.concurrent.CountDownLatch;

/** Implementation of {@link FsEventsNativeDepsService}. */
public class FsEventsNativeDepsServiceImpl implements FsEventsNativeDepsService {
  private static final boolean JNI_AVAILABLE;

  static {
    boolean loadJniWorked = false;
    try {
      JniLoader.loadJni();
      loadJniWorked = true;
    } catch (UnsatisfiedLinkError ignored) {
      // Unfortunately, we compile this class into the Bazel bootstrap binary, which doesn't have
      // access to the JNI code (to simplify bootstrap). This is the quick and dirty way to
      // hard-disable --watchfs in the bootstrap binary.
    }
    JNI_AVAILABLE = loadJniWorked;
  }

  // Keep a pointer to a native structure in the JNI code (the FsEvents callback needs that
  // structure).
  private long nativePointer;

  @Override
  public boolean isJniAvailable() {
    return JNI_AVAILABLE;
  }

  @Override
  public void createFsEvents(byte[][] paths, byte[][] excludedPaths, double latency) {
    create(paths, excludedPaths, latency);
  }

  @Override
  public void runFsEvents(CountDownLatch listening) {
    run(listening);
  }

  @Override
  public void doCloseFsEvents() {
    doClose();
  }

  @Override
  public byte[][] pollFsEvents() {
    return poll();
  }

  private native void create(byte[][] paths, byte[][] excludedPaths, double latency);

  private native void run(CountDownLatch listening);

  private native void doClose();

  private native byte[][] poll();
}
