// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import sun.misc.Signal;
import sun.misc.SignalHandler;

/**
 * A facade around sun.misc.Signal providing special-purpose SIGINT handling.
 *
 * <p>We use this code in preference to using sun.misc directly since the latter is deprecated, and
 * depending on it causes the jdk1.6 javac to emit an unsuppressable warning that sun.misc is
 * "Sun proprietary API and may be removed in a future release".
 */
public abstract class InterruptSignalHandler implements Runnable {

  private static final Signal SIGINT = new Signal("INT");

  private SignalHandler oldHandler;

  /**
   * Constructs an InterruptSignalHandler instance.  Until the uninstall()
   * method is invoked, the delivery of a SIGINT signal to this process will
   * cause the run() method to be invoked in another thread.
   */
  protected InterruptSignalHandler() {
    this.oldHandler = Signal.handle(SIGINT, signal -> run());
  }

  /** Disables SIGINT handling. */
  public final synchronized void uninstall() {
    Preconditions.checkNotNull(oldHandler, "uninstall() already called");
    Signal.handle(SIGINT, oldHandler);
    oldHandler = null;
  }
}
