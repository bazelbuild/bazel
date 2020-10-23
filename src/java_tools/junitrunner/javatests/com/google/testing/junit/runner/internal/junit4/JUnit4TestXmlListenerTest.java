// Copyright 2011 The Bazel Authors. All Rights Reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.fail;
import static org.junit.Assume.assumeTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.Lists;
import com.google.testing.junit.runner.internal.SignalHandlers;
import com.google.testing.junit.runner.model.TestSuiteModel;
import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.List;
import java.util.function.Supplier;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.Description;
import org.junit.runner.JUnitCore;
import org.junit.runner.Request;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.InOrder;
import sun.misc.Signal;
import sun.misc.SignalHandler;

/**
 * Tests for {@link JUnit4TestXmlListener}
 */
@RunWith(JUnit4.class)
public class JUnit4TestXmlListenerTest {
  private static final String CHARSET = "UTF-8";
  private final FakeSignalHandlers fakeSignalHandlers = new FakeSignalHandlers();
  private final ByteArrayOutputStream errStream = new ByteArrayOutputStream();
  private PrintStream errPrintStream;

  @Before
  public void createErrPrintStream() throws Exception {
    errPrintStream = new PrintStream(errStream, true, CHARSET);
  }

  @Test
  public void signalHandlerWritesXml() throws Exception {
    TestSuiteModelSupplier mockModelSupplier = mock(TestSuiteModelSupplier.class);
    TestSuiteModel mockModel = mock(TestSuiteModel.class);
    CancellableRequestFactory mockRequestFactory = mock(CancellableRequestFactory.class);
    OutputStream mockXmlStream = mock(OutputStream.class);
    JUnit4TestXmlListener listener = new JUnit4TestXmlListener(
        mockModelSupplier, mockRequestFactory, fakeSignalHandlers, mockXmlStream, errPrintStream);

    Request request = Request.classWithoutSuiteMethod(PassingTest.class);
    Description suiteDescription = request.getRunner().getDescription();

    when(mockModelSupplier.get()).thenReturn(mockModel);

    listener.testRunStarted(suiteDescription);
    assertThat(fakeSignalHandlers.handlers).hasSize(1);

    fakeSignalHandlers.handlers.get(0).handle(new Signal("TERM"));

    String errOutput = errStream.toString(CHARSET);
    assertWithMessage("expected signal name in stderr")
        .that(errOutput.contains("SIGTERM"))
        .isTrue();
    assertWithMessage("expected message in stderr")
        .that(errOutput.contains("Done writing test XML"))
        .isTrue();

    InOrder inOrder = inOrder(mockRequestFactory, mockModel);
    inOrder.verify(mockRequestFactory).cancelRun();
    inOrder.verify(mockModel).testRunInterrupted();
    inOrder.verify(mockModel).writeAsXml(mockXmlStream);
  }

  @Test
  public void writesXmlAtTestEnd() throws Exception {
    TestSuiteModelSupplier mockModelSupplier = mock(TestSuiteModelSupplier.class);
    TestSuiteModel mockModel = mock(TestSuiteModel.class);
    CancellableRequestFactory mockRequestFactory = mock(CancellableRequestFactory.class);
    OutputStream mockXmlStream = mock(OutputStream.class);
    JUnit4TestXmlListener listener = new JUnit4TestXmlListener(
        mockModelSupplier, mockRequestFactory, fakeSignalHandlers, mockXmlStream, errPrintStream);

    when(mockModelSupplier.get()).thenReturn(mockModel);

    JUnitCore core = new JUnitCore();
    core.addListener(listener);
    core.run(Request.classWithoutSuiteMethod(PassingTest.class));

    assertWithMessage("no output to stderr expected").that(errStream.size()).isEqualTo(0);
    verify(mockModel).writeAsXml(mockXmlStream);
    verify(mockRequestFactory, never()).cancelRun();
  }

  @Test
  public void assumptionViolationsAreReportedAsSkippedTests() throws Exception {
    TestSuiteModelSupplier mockModelSupplier = mock(TestSuiteModelSupplier.class);
    TestSuiteModel mockModel = mock(TestSuiteModel.class);
    CancellableRequestFactory mockRequestFactory = mock(CancellableRequestFactory.class);
    OutputStream mockXmlStream = mock(OutputStream.class);
    JUnit4TestXmlListener listener = new JUnit4TestXmlListener(
        mockModelSupplier, mockRequestFactory, fakeSignalHandlers, mockXmlStream, errPrintStream);

    Request request = Request.classWithoutSuiteMethod(TestWithAssumptionViolation.class);
    Description suiteDescription = request.getRunner().getDescription();
    Description testDescription = suiteDescription.getChildren().get(0);

    when(mockModelSupplier.get()).thenReturn(mockModel);

    JUnitCore core = new JUnitCore();
    core.addListener(listener);
    core.run(request);

    assertWithMessage("no output to stderr expected").that(errStream.size()).isEqualTo(0);
    InOrder inOrder = inOrder(mockModel);
    inOrder.verify(mockModel).testSkipped(testDescription);
    inOrder.verify(mockModel).writeAsXml(mockXmlStream);
    verify(mockRequestFactory, never()).cancelRun();
  }

  @Test
  public void assumptionViolationsAtSuiteLevelAreReportedAsSkippedSuite() throws Exception {
    TestSuiteModelSupplier mockModelSupplier = mock(TestSuiteModelSupplier.class);
    TestSuiteModel mockModel = mock(TestSuiteModel.class);
    CancellableRequestFactory mockRequestFactory = mock(CancellableRequestFactory.class);
    OutputStream mockXmlStream = mock(OutputStream.class);
    JUnit4TestXmlListener listener = new JUnit4TestXmlListener(
        mockModelSupplier, mockRequestFactory, fakeSignalHandlers, mockXmlStream, errPrintStream);

    Request request = Request.classWithoutSuiteMethod(
        TestWithAssumptionViolationOnTheSuiteLevel.class);
    Description suiteDescription = request.getRunner().getDescription();

    when(mockModelSupplier.get()).thenReturn(mockModel);

    JUnitCore core = new JUnitCore();
    core.addListener(listener);
    core.run(request);

    assertWithMessage("no output to stderr expected").that(errStream.size()).isEqualTo(0);
    InOrder inOrder = inOrder(mockModel);
    inOrder.verify(mockModel).testSkipped(suiteDescription);
    inOrder.verify(mockModel).writeAsXml(mockXmlStream);
    verify(mockRequestFactory, never()).cancelRun();
  }

  @Test
  public void failuresAreReported() throws Exception {
    TestSuiteModelSupplier mockModelSupplier = mock(TestSuiteModelSupplier.class);
    TestSuiteModel mockModel = mock(TestSuiteModel.class);
    CancellableRequestFactory mockRequestFactory = mock(CancellableRequestFactory.class);
    OutputStream mockXmlStream = mock(OutputStream.class);
    JUnit4TestXmlListener listener = new JUnit4TestXmlListener(
        mockModelSupplier, mockRequestFactory, fakeSignalHandlers, mockXmlStream, errPrintStream);

    Request request = Request.classWithoutSuiteMethod(FailingTest.class);
    Description suiteDescription = request.getRunner().getDescription();
    Description testDescription = suiteDescription.getChildren().get(0);

    when(mockModelSupplier.get()).thenReturn(mockModel);

    JUnitCore core = new JUnitCore();
    core.addListener(listener);
    core.run(request);

    assertWithMessage("no output to stderr expected").that(errStream.size()).isEqualTo(0);
    InOrder inOrder = inOrder(mockModel);
    inOrder.verify(mockModel).testFailure(eq(testDescription), any(Throwable.class));
    inOrder.verify(mockModel).writeAsXml(mockXmlStream);
    verify(mockRequestFactory, never()).cancelRun();
  }

  @Test
  public void ignoredTestAreReportedAsSuppressedTests() throws Exception {
    TestSuiteModelSupplier mockModelSupplier = mock(TestSuiteModelSupplier.class);
    TestSuiteModel mockModel = mock(TestSuiteModel.class);
    CancellableRequestFactory mockRequestFactory = mock(CancellableRequestFactory.class);
    OutputStream mockXmlStream = mock(OutputStream.class);
    JUnit4TestXmlListener listener = new JUnit4TestXmlListener(
        mockModelSupplier, mockRequestFactory, fakeSignalHandlers, mockXmlStream, errPrintStream);

    Request request = Request.classWithoutSuiteMethod(TestWithIgnoredTestCase.class);
    Description suiteDescription = request.getRunner().getDescription();
    Description testDescription = suiteDescription.getChildren().get(0);

    when(mockModelSupplier.get()).thenReturn(mockModel);

    JUnitCore core = new JUnitCore();
    core.addListener(listener);
    core.run(request);

    assertWithMessage("no output to stderr expected").that(errStream.size()).isEqualTo(0);
    InOrder inOrder = inOrder(mockModel);
    inOrder.verify(mockModel).testSuppressed(testDescription);
    inOrder.verify(mockModel).writeAsXml(mockXmlStream);
    verify(mockRequestFactory, never()).cancelRun();
  }

  /**
   * Test with a method that always passes.
   */
  public static class PassingTest {
    @Test
    public void alwaysPasses() {
    }
  }


  /**
   * Test with a method that always fails.
   */
  public static class FailingTest {
    @Test
    public void alwaysFails() {
      fail();
    }
  }


  /**
   * Test with a method that is always skipped.
   */
  public static class TestWithAssumptionViolation {
    @Test
    public void alwaysSkipped() {
      assumeTrue(false);
    }
  }

  /**
   * Test with a method that is always skipped.
   */
  public static class TestWithAssumptionViolationOnTheSuiteLevel {
    @BeforeClass
    public static void failedSuiteLevelAssumption() {
      assumeTrue(false);
    }

    @Test
    public void fakeTest() {}
  }

  public static class TestWithIgnoredTestCase {
    @Test
    @Ignore
    public void alwaysIgnored() {

    }
  }

  private static class FakeSignalHandlers extends SignalHandlers {
    public List<SignalHandler> handlers = Lists.newArrayList();

    public FakeSignalHandlers() {
      super(null);
    }

    @Override
    public void installHandler(Signal signal, SignalHandler signalHandler) {
      if (signal.getName().equals("TERM")) {
        handlers.add(signalHandler);
      }
    }
  }


  private interface TestSuiteModelSupplier extends Supplier<TestSuiteModel> {
  }
}