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
package com.google.devtools.build.lib.concurrent.safeexecutor;

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;

/** Task interface extending Runnable to handle both execution and submission failures directly. */
@SkybridgeInterface
public interface RejectionHandlingRunnable extends Runnable {

  /**
   * Invoked when task submission fails (rejection).
   *
   * <p>Implementations MUST be non-blocking.
   */
  void handleRejection(Throwable t);
}
