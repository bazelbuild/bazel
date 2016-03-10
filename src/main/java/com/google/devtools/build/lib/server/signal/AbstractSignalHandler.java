// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.server.signal;


import com.google.devtools.build.lib.util.Preconditions;

import sun.misc.Signal;
import sun.misc.SignalHandler;

/**
 * A facade around {@link sun.misc.Signal} providing special-purpose signal handling.
 *
 * <p>We use this code in preference to using sun.misc directly since the latter is deprecated, and
 * depending on it causes javac to emit an unsuppressable warning that sun.misc is
 * "Sun proprietary API and may be removed in a future release".
 */
public abstract class AbstractSignalHandler {
  private final Signal signal;
  private SignalHandler oldHandler;

  /**
   * Constructs an AbstractSignalHandler instance.  Until the uninstall() method is invoked, the
   * delivery of {@code signal} to this process will cause the run() method to be invoked in another
   * thread.
   */
  protected AbstractSignalHandler(Signal signal) {
    this.signal = signal;
    this.oldHandler =
        Signal.handle(
            signal,
            new SignalHandler() {
              @Override
              public void handle(Signal signal) {
                onSignal();
              }
            });
  }

  protected abstract void onSignal();

  /** Disables signal handling. */
  public final synchronized void uninstall() {
    Preconditions.checkNotNull(oldHandler, "uninstall() already called");
    Signal.handle(signal, oldHandler);
    oldHandler = null;
  }
}
