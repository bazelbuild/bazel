// Copyright 2012 The Bazel Authors. All Rights Reserved.
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

import com.google.common.base.Supplier;
import com.google.inject.Inject;
import com.google.inject.Singleton;
import com.google.testing.junit.runner.internal.SignalHandlers;
import com.google.testing.junit.runner.internal.Stderr;
import com.google.testing.junit.runner.model.TestSuiteModel;

import org.junit.Ignore;
import org.junit.runner.Description;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;
import org.junit.runner.notification.RunListener;

import sun.misc.Signal;
import sun.misc.SignalHandler;

import java.io.OutputStream;
import java.io.PrintStream;

/**
 * A listener that writes the test output as XML.
 */
@Singleton
class JUnit4TestXmlListener extends RunListener  {
  
  private final Supplier<TestSuiteModel> modelSupplier;
  private final CancellableRequestFactory requestFactory;
  private final SignalHandlers signalHandlers;
  private final OutputStream xmlStream;
  private final PrintStream errPrintStream;
  private volatile TestSuiteModel model;

  @Inject
  public JUnit4TestXmlListener(Supplier<TestSuiteModel> modelSupplier,
      CancellableRequestFactory requestFactory, SignalHandlers signalHandlers,
      @Xml OutputStream xmlStream, @Stderr PrintStream errPrintStream) {
    this.modelSupplier = modelSupplier;
    this.requestFactory = requestFactory;
    this.signalHandlers = signalHandlers;
    this.xmlStream = xmlStream;
    this.errPrintStream = errPrintStream;
  }

  @Override
  public void testRunStarted(Description description) throws Exception {
    model = modelSupplier.get();
    signalHandlers.installHandler(new Signal("TERM"), new WriteXmlSignalHandler());
  }

  @Override
  public void testStarted(Description description) throws Exception {
    model.testStarted(description);
  }

  @Override
  public void testAssumptionFailure(Failure failure) {
    model.testSkipped(failure.getDescription());
  }

  @Override
  public void testFailure(Failure failure) throws Exception {
    model.testFailure(failure.getDescription(), failure.getException());
  }

  @Override
  public void testIgnored(Description description) throws Exception {
    // TODO(bazel-team) There's a known issue in the JUnit4 ParentRunner that
    // fires testIgnored on test suites that are being skipped due to an
    // assumption failure.
    if (isSuiteAssumptionFailure(description)) {
      model.testSkipped(description);
    } else {
      model.testSuppressed(description);
    }
  }

  private boolean isSuiteAssumptionFailure(Description description) {
    return description.isSuite() && description.getAnnotation(Ignore.class) == null;
  }

  @Override
  public void testFinished(Description description) throws Exception {
    model.testFinished(description);
  }

  @Override
  public void testRunFinished(Result result) throws Exception {
    model.writeAsXml(xmlStream);
  }

  private class WriteXmlSignalHandler implements SignalHandler {

    @Override
    public void handle(Signal signal) {
      try {
        errPrintStream.printf("%nReceived %s; writing test XML%n", signal.toString());
        requestFactory.cancelRun();
        model.testRunInterrupted();
        model.writeAsXml(xmlStream);
        errPrintStream.println("Done writing test XML");
      } catch (Exception e) {
        errPrintStream.println("Could not write test XML");
        e.printStackTrace(errPrintStream);
      }
    }
  }
}
