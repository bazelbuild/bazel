// Copyright 2015 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.internal.junit4;

import com.google.testing.junit.runner.internal.SignalHandlers;
import com.google.testing.junit.runner.internal.StackTraces;
import java.io.PrintStream;
import org.junit.runner.Description;
import org.junit.runner.notification.RunListener;
import sun.misc.Signal;
import sun.misc.SignalHandler;

/** A listener than dumps all stack traces when the test receives a SIGTERM. */
@SuppressWarnings("SunApi") // no alternative for signal handling?
public class JUnit4TestStackTraceListener extends RunListener {
  private final SignalHandlers signalHandlers;
  private final PrintStream errPrintStream;

  public JUnit4TestStackTraceListener(SignalHandlers signalHandlers, PrintStream errPrintStream) {
    this.signalHandlers = signalHandlers;
    this.errPrintStream = errPrintStream;
  }

  @Override
  public void testRunStarted(Description description) throws Exception {
    signalHandlers.installHandler(new Signal("TERM"), new WriteStackTraceSignalHandler());
  }

  private class WriteStackTraceSignalHandler implements SignalHandler {
    @Override
    public void handle(Signal signal) {
      errPrintStream.println("Dumping stack traces for all threads\n");
      StackTraces.printAll(errPrintStream);
    }
  }
}
