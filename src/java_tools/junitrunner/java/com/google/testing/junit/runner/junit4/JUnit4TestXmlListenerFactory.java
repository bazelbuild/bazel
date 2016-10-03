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
import com.google.testing.junit.runner.internal.junit4.CancellableRequestFactory;
import com.google.testing.junit.runner.internal.junit4.JUnit4TestXmlListener;
import com.google.testing.junit.runner.model.TestSuiteModel;
import com.google.testing.junit.runner.util.Factory;
import com.google.testing.junit.runner.util.Supplier;
import java.io.OutputStream;
import java.io.PrintStream;

/**
 * A factory that supplies {@link JUnit4TestXmlListener}.
 */
public final class JUnit4TestXmlListenerFactory implements Factory<JUnit4TestXmlListener> {
  private final Supplier<Supplier<TestSuiteModel>> modelSupplierSupplier;

  private final Supplier<CancellableRequestFactory> requestFactorySupplier;

  private final Supplier<SignalHandlers> signalHandlersSupplier;

  private final Supplier<OutputStream> xmlStreamSupplier;

  private final Supplier<PrintStream> errPrintStreamSupplier;

  public JUnit4TestXmlListenerFactory(
      Supplier<Supplier<TestSuiteModel>> modelSupplierSupplier,
      Supplier<CancellableRequestFactory> requestFactorySupplier,
      Supplier<SignalHandlers> signalHandlersSupplier,
      Supplier<OutputStream> xmlStreamSupplier,
      Supplier<PrintStream> errPrintStreamSupplier) {
    assert modelSupplierSupplier != null;
    this.modelSupplierSupplier = modelSupplierSupplier;
    assert requestFactorySupplier != null;
    this.requestFactorySupplier = requestFactorySupplier;
    assert signalHandlersSupplier != null;
    this.signalHandlersSupplier = signalHandlersSupplier;
    assert xmlStreamSupplier != null;
    this.xmlStreamSupplier = xmlStreamSupplier;
    assert errPrintStreamSupplier != null;
    this.errPrintStreamSupplier = errPrintStreamSupplier;
  }

  @Override
  public JUnit4TestXmlListener get() {
      return new JUnit4TestXmlListener(
            modelSupplierSupplier.get(),
            requestFactorySupplier.get(),
            signalHandlersSupplier.get(),
            xmlStreamSupplier.get(),
            errPrintStreamSupplier.get());
  }

  public static Factory<JUnit4TestXmlListener> create(
      Supplier<Supplier<TestSuiteModel>> modelSupplierSupplier,
      Supplier<CancellableRequestFactory> requestFactorySupplier,
      Supplier<SignalHandlers> signalHandlersSupplier,
      Supplier<OutputStream> xmlStreamSupplier,
      Supplier<PrintStream> errPrintStreamSupplier) {
    return new JUnit4TestXmlListenerFactory(
        modelSupplierSupplier,
        requestFactorySupplier,
        signalHandlersSupplier,
        xmlStreamSupplier,
        errPrintStreamSupplier);
  }
}
