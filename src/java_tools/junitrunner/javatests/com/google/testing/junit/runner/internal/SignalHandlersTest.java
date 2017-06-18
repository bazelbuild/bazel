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

import static com.google.common.truth.Truth.assertWithMessage;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;
import sun.misc.Signal;
import sun.misc.SignalHandler;

/**
 * Tests for SignalHandlers.
 */
@RunWith(JUnit4.class)
public class SignalHandlersTest {

  private static final Signal TERM_SIGNAL = new Signal("TERM");

  private final FakeSignalInstaller fakeSignalInstaller = new FakeSignalInstaller();
  private final SignalHandlers signalHandlers = new SignalHandlers(fakeSignalInstaller);

  static class FakeSignalInstaller implements SignalHandlers.HandlerInstaller {
    private SignalHandler currentHandler = null;

    @Override
    public SignalHandler install(Signal signal, SignalHandler handler) {
      SignalHandler previousHandler = currentHandler;
      assertWithMessage("This fake only supports the TERM signal")
          .that(signal)
          .isEqualTo(TERM_SIGNAL);
      currentHandler = handler;
      return previousHandler;
    }

    public void sendSignal() {
      currentHandler.handle(TERM_SIGNAL);
    }
  }

  @Test
  public void testHandlersCanBeChained() {
    SignalHandler handler1 = Mockito.mock(SignalHandler.class);
    SignalHandler handler2 = Mockito.mock(SignalHandler.class);

    signalHandlers.installHandler(TERM_SIGNAL, handler1);
    signalHandlers.installHandler(TERM_SIGNAL, handler2);
    fakeSignalInstaller.sendSignal();

    Mockito.verify(handler1).handle(Mockito.eq(TERM_SIGNAL));
    Mockito.verify(handler2).handle(Mockito.eq(TERM_SIGNAL));
  }

  @Test
  public void testOneHandlerCanHandleSignal() {
    SignalHandler handler = Mockito.mock(SignalHandler.class);

    signalHandlers.installHandler(TERM_SIGNAL, handler);
    fakeSignalInstaller.sendSignal();
    
    Mockito.verify(handler).handle(Mockito.eq(TERM_SIGNAL));
  }
}
