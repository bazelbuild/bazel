// Copyright 2010 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.internal;

import java.util.concurrent.atomic.AtomicReference;
import sun.misc.Signal;
import sun.misc.SignalHandler;

/** Helper class to install signal handlers. */
@SuppressWarnings("SunApi") // no alternative for signal handling?
public class SignalHandlers {
  private final HandlerInstaller handlerInstaller;

  /**
   * Creates a handler installer that installs signal handlers.
   */
  public static HandlerInstaller createRealHandlerInstaller() {
    return new HandlerInstaller() {
      @Override
      public SignalHandler install(Signal signal, SignalHandler handler) {
        return Signal.handle(signal, handler);
      }
    };
  }

  public SignalHandlers(HandlerInstaller installer) {
    this.handlerInstaller = installer;
  }

  /**
   * Adds the given signal handler to the existing ones.
   *
   * <p>Signal handlers are responsible to catch any exception if the following
   * handlers need to be executed when a handler throws an exception.
   *
   * @param signal The signal to handle.
   * @param signalHandler The handler to install.
   */
  public void installHandler(Signal signal, final SignalHandler signalHandler) {
    final AtomicReference<SignalHandler> previousHandlerReference =
        new AtomicReference<>();
    previousHandlerReference.set(handlerInstaller.install(signal, new SignalHandler() {
      @Override
      public void handle(Signal signal) {
        signalHandler.handle(signal);
        SignalHandler previousHandler = previousHandlerReference.get();
        if (previousHandler != null) {
          previousHandler.handle(signal);
        }
      }
    }));
  }

  /**
   * Wraps sun.misc.Signal#handle(sun.misc.Signal, sun.misc.SignalHandler)
   * to help with testing.
   */
  public interface HandlerInstaller {
    /**
     * @see sun.misc.Signal#handle(sun.misc.Signal, sun.misc.SignalHandler)
     */
    SignalHandler install(Signal signal, SignalHandler handler);
  }
}
