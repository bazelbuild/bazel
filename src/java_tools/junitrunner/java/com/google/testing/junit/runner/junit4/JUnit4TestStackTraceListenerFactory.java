// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.junit4;

import com.google.testing.junit.runner.internal.SignalHandlers;
import com.google.testing.junit.runner.internal.junit4.JUnit4TestStackTraceListener;
import com.google.testing.junit.runner.util.Factory;
import com.google.testing.junit.runner.util.Supplier;
import java.io.PrintStream;

/**
 * A factory that supplies {@link JUnit4TestStackTraceListener}.
 */
public final class JUnit4TestStackTraceListenerFactory
    implements Factory<JUnit4TestStackTraceListener> {

  private final Supplier<SignalHandlers> signalHandlersSupplier;

  private final Supplier<PrintStream> errPrintStreamSupplier;

  public JUnit4TestStackTraceListenerFactory(
      Supplier<SignalHandlers> signalHandlersSupplier,
      Supplier<PrintStream> errPrintStreamSupplier) {
    assert signalHandlersSupplier != null;
    this.signalHandlersSupplier = signalHandlersSupplier;
    assert errPrintStreamSupplier != null;
    this.errPrintStreamSupplier = errPrintStreamSupplier;
  }

  @Override
  public JUnit4TestStackTraceListener get() {
    return new JUnit4TestStackTraceListener(
        signalHandlersSupplier.get(), errPrintStreamSupplier.get());
  }

  public static Factory<JUnit4TestStackTraceListener> create(
      Supplier<SignalHandlers> signalHandlersSupplier,
      Supplier<PrintStream> errPrintStreamSupplier) {
    return new JUnit4TestStackTraceListenerFactory(signalHandlersSupplier, errPrintStreamSupplier);
  }
}
