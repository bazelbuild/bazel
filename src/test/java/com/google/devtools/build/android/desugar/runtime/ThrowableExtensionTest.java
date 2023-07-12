// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar.runtime;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtension.MimicDesugaringStrategy.SUPPRESSED_PREFIX;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTestUtility.getTwrStrategyClassNameSpecifiedInSystemProperty;
import static com.google.devtools.build.android.desugar.runtime.ThrowableExtensionTestUtility.isNullStrategy;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.devtools.build.android.desugar.runtime.ThrowableExtension.MimicDesugaringStrategy;
import com.google.devtools.build.android.desugar.runtime.ThrowableExtension.NullDesugaringStrategy;
import com.google.devtools.build.android.desugar.runtime.ThrowableExtension.ReuseDesugaringStrategy;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.function.Consumer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for {@link ThrowableExtension} */
@RunWith(JUnit4.class)
public class ThrowableExtensionTest {

  /**
   * This test tests the behavior of closing resources via reflection. This is only enabled below
   * API 19. So, if the API level is 19 or above, this test will simply skip.
   */
  @Test
  public void testCloseResourceViaReflection() throws Throwable {
    class Resource extends AbstractResource {
      protected Resource(boolean exceptionOnClose) {
        super(exceptionOnClose);
      }

      public void close() throws Exception {
        super.internalClose();
      }
    }
    if (ThrowableExtension.API_LEVEL >= 19) {
      return;
    }
    {
      Resource r = new Resource(false);
      assertThat(r.isClosed()).isFalse();
      ThrowableExtension.closeResource(null, r);
      assertThat(r.isClosed()).isTrue();
    }
    {
      Resource r = new Resource(true);
      assertThat(r.isClosed()).isFalse();
      assertThrows(IOException.class, () -> ThrowableExtension.closeResource(null, r));
    }
    {
      Resource r = new Resource(false);
      assertThat(r.isClosed()).isFalse();
      ThrowableExtension.closeResource(new Exception(), r);
      assertThat(r.isClosed()).isTrue();
    }
    {
      Resource r = new Resource(true);
      assertThat(r.isClosed()).isFalse();
      assertThrows(Exception.class, () -> ThrowableExtension.closeResource(new Exception(), r));
    }
  }

  /**
   * Test the new method closeResources() in the runtime library.
   *
   * <p>The method is introduced to fix b/37167433.
   */
  @Test
  public void testCloseResource() throws Throwable {

    /**
     * A resource implementing the interface AutoCloseable. This interface is only available since
     * API 19.
     */
    class AutoCloseableResource extends AbstractResource implements AutoCloseable {

      protected AutoCloseableResource(boolean exceptionOnClose) {
        super(exceptionOnClose);
      }

      @Override
      public void close() throws Exception {
        internalClose();
      }
    }

    /** A resource implementing the interface Closeable. */
    class CloseableResource extends AbstractResource implements Closeable {

      protected CloseableResource(boolean exceptionOnClose) {
        super(exceptionOnClose);
      }

      @Override
      public void close() throws IOException {
        internalClose();
      }
    }

    {
      CloseableResource r = new CloseableResource(false);
      assertThat(r.isClosed()).isFalse();
      ThrowableExtension.closeResource(null, r);
      assertThat(r.isClosed()).isTrue();
    }
    {
      CloseableResource r = new CloseableResource(false);
      assertThat(r.isClosed()).isFalse();
      Exception suppressor = new Exception();
      ThrowableExtension.closeResource(suppressor, r);
      assertThat(r.isClosed()).isTrue();
      assertThat(ThrowableExtension.getSuppressed(suppressor)).isEmpty();
    }
    {
      CloseableResource r = new CloseableResource(true);
      assertThat(r.isClosed()).isFalse();
      assertThrows(IOException.class, () -> ThrowableExtension.closeResource(null, r));
      assertThat(r.isClosed()).isFalse();
    }
    {
      CloseableResource r = new CloseableResource(true);
      assertThat(r.isClosed()).isFalse();
      Exception suppressor = new Exception();
      assertThrows(Exception.class, () -> ThrowableExtension.closeResource(suppressor, r));
      assertThat(r.isClosed()).isFalse(); // Failed to close.
      if (!isNullStrategy()) {
        assertThat(ThrowableExtension.getSuppressed(suppressor)).hasLength(1);
        assertThat(ThrowableExtension.getSuppressed(suppressor)[0].getClass())
            .isEqualTo(IOException.class);
      }
    }
    {
      AutoCloseableResource r = new AutoCloseableResource(false);
      assertThat(r.isClosed()).isFalse();
      ThrowableExtension.closeResource(null, r);
      assertThat(r.isClosed()).isTrue();
    }
    {
      AutoCloseableResource r = new AutoCloseableResource(false);
      assertThat(r.isClosed()).isFalse();
      Exception suppressor = new Exception();
      ThrowableExtension.closeResource(suppressor, r);
      assertThat(r.isClosed()).isTrue();
      assertThat(ThrowableExtension.getSuppressed(suppressor)).isEmpty();
    }
    {
      AutoCloseableResource r = new AutoCloseableResource(true);
      assertThat(r.isClosed()).isFalse();
      assertThrows(IOException.class, () -> ThrowableExtension.closeResource(null, r));
      assertThat(r.isClosed()).isFalse();
    }
    {
      AutoCloseableResource r = new AutoCloseableResource(true);
      assertThat(r.isClosed()).isFalse();
      Exception suppressor = new Exception();
      assertThrows(Exception.class, () -> ThrowableExtension.closeResource(suppressor, r));
      assertThat(r.isClosed()).isFalse(); // Failed to close.
      if (!isNullStrategy()) {
        assertThat(ThrowableExtension.getSuppressed(suppressor)).hasLength(1);
        assertThat(ThrowableExtension.getSuppressed(suppressor)[0].getClass())
            .isEqualTo(IOException.class);
      }
      assertThat(r.isClosed()).isFalse();
    }
  }

  /**
   * LightweightStackTraceRecorder tracks the calls of various printStackTrace(*), and ensures that
   *
   * <p>suppressed exceptions are printed only once.
   */
  @Test
  public void testLightweightStackTraceRecorder() throws IOException {
    MimicDesugaringStrategy strategy = new MimicDesugaringStrategy();
    ExceptionForTest receiver = new ExceptionForTest(strategy);
    FileNotFoundException suppressed = new FileNotFoundException();
    strategy.addSuppressed(receiver, suppressed);

    String trace = printStackTraceStderrToString(() -> strategy.printStackTrace(receiver));
    assertThat(trace).contains(SUPPRESSED_PREFIX);
    assertThat(countOccurrences(trace, SUPPRESSED_PREFIX)).isEqualTo(1);
  }

  @Test
  public void testMimicDesugaringStrategy() throws IOException {
    MimicDesugaringStrategy strategy = new MimicDesugaringStrategy();
    IOException receiver = new IOException();
    FileNotFoundException suppressed = new FileNotFoundException();
    strategy.addSuppressed(receiver, suppressed);

    assertThat(
            printStackTracePrintStreamToString(
                stream -> strategy.printStackTrace(receiver, stream)))
        .contains(SUPPRESSED_PREFIX);

    assertThat(
            printStackTracePrintWriterToString(
                writer -> strategy.printStackTrace(receiver, writer)))
        .contains(SUPPRESSED_PREFIX);

    assertThat(printStackTraceStderrToString(() -> strategy.printStackTrace(receiver)))
        .contains(SUPPRESSED_PREFIX);
  }

  private void testThrowableExtensionWithMimicDesugaringStrategy() throws IOException {
    IOException receiver = new IOException();
    FileNotFoundException suppressed = new FileNotFoundException();
    ThrowableExtension.addSuppressed(receiver, suppressed);

    assertThat(
            printStackTracePrintStreamToString(
                stream -> ThrowableExtension.printStackTrace(receiver, stream)))
        .contains(SUPPRESSED_PREFIX);
    assertThat(
            printStackTracePrintWriterToString(
                writer -> ThrowableExtension.printStackTrace(receiver, writer)))
        .contains(SUPPRESSED_PREFIX);
    assertThat(printStackTraceStderrToString(() -> ThrowableExtension.printStackTrace(receiver)))
        .contains(SUPPRESSED_PREFIX);
  }

  private interface PrintStackTraceCaller {
    void printStackTrace();
  }

  private static String printStackTraceStderrToString(PrintStackTraceCaller caller)
      throws IOException {
    PrintStream err = System.err;
    try (ByteArrayOutputStream stream = new ByteArrayOutputStream()) {
      PrintStream newErr = new PrintStream(stream);
      System.setErr(newErr);
      caller.printStackTrace();
      newErr.flush();
      return stream.toString();
    } finally {
      System.setErr(err);
    }
  }

  private static String printStackTracePrintStreamToString(Consumer<PrintStream> caller)
      throws IOException {
    try (ByteArrayOutputStream stream = new ByteArrayOutputStream()) {
      PrintStream printStream = new PrintStream(stream);
      caller.accept(printStream);
      printStream.flush();
      return stream.toString();
    }
  }

  private static String printStackTracePrintWriterToString(Consumer<PrintWriter> caller)
      throws IOException {
    try (ByteArrayOutputStream stream = new ByteArrayOutputStream()) {
      PrintWriter printWriter =
          new PrintWriter(new BufferedWriter(new OutputStreamWriter(stream, UTF_8)));
      caller.accept(printWriter);
      printWriter.flush();
      return stream.toString();
    }
  }

  @Test
  public void testNullDesugaringStrategy() throws IOException {
    NullDesugaringStrategy strategy = new NullDesugaringStrategy();
    IOException receiver = new IOException();
    FileNotFoundException suppressed = new FileNotFoundException();
    strategy.addSuppressed(receiver, suppressed);
    assertThat(strategy.getSuppressed(receiver)).isEmpty();

    strategy.addSuppressed(receiver, suppressed);
    assertThat(strategy.getSuppressed(receiver)).isEmpty();

    assertThat(printStackTracePrintStreamToString(stream -> receiver.printStackTrace(stream)))
        .isEqualTo(
            printStackTracePrintStreamToString(
                stream -> strategy.printStackTrace(receiver, stream)));

    assertThat(printStackTracePrintWriterToString(receiver::printStackTrace))
        .isEqualTo(
            printStackTracePrintWriterToString(
                writer -> strategy.printStackTrace(receiver, writer)));

    assertThat(printStackTraceStderrToString(receiver::printStackTrace))
        .isEqualTo(printStackTraceStderrToString(() -> strategy.printStackTrace(receiver)));
  }

  private void testThrowableExtensionWithNullDesugaringStrategy() throws IOException {
    IOException receiver = new IOException();
    FileNotFoundException suppressed = new FileNotFoundException();
    ThrowableExtension.addSuppressed(receiver, suppressed);
    assertThat(ThrowableExtension.getSuppressed(receiver)).isEmpty();

    ThrowableExtension.addSuppressed(receiver, suppressed);
    assertThat(ThrowableExtension.getSuppressed(receiver)).isEmpty();

    assertThat(printStackTracePrintStreamToString(stream -> receiver.printStackTrace(stream)))
        .isEqualTo(
            printStackTracePrintStreamToString(
                stream -> ThrowableExtension.printStackTrace(receiver, stream)));
    assertThat(printStackTracePrintWriterToString(receiver::printStackTrace))
        .isEqualTo(
            printStackTracePrintWriterToString(
                writer -> ThrowableExtension.printStackTrace(receiver, writer)));

    assertThat(printStackTraceStderrToString(receiver::printStackTrace))
        .isEqualTo(
            printStackTraceStderrToString(() -> ThrowableExtension.printStackTrace(receiver)));
  }

  @Test
  public void testReuseDesugaringStrategy() throws IOException {
    ReuseDesugaringStrategy strategy = new ReuseDesugaringStrategy();
    IOException receiver = new IOException();
    FileNotFoundException suppressed = new FileNotFoundException();
    strategy.addSuppressed(receiver, suppressed);
    assertThat(strategy.getSuppressed(receiver))
        .asList()
        .containsExactly((Object[]) receiver.getSuppressed());

    assertThat(printStackTracePrintStreamToString(stream -> receiver.printStackTrace(stream)))
        .isEqualTo(
            printStackTracePrintStreamToString(
                stream -> strategy.printStackTrace(receiver, stream)));

    assertThat(printStackTracePrintWriterToString(receiver::printStackTrace))
        .isEqualTo(
            printStackTracePrintWriterToString(
                writer -> strategy.printStackTrace(receiver, writer)));
    assertThat(printStackTraceStderrToString(receiver::printStackTrace))
        .isEqualTo(printStackTraceStderrToString(() -> strategy.printStackTrace(receiver)));
  }

  private void testThrowableExtensionWithReuseDesugaringStrategy() throws IOException {
    IOException receiver = new IOException();
    FileNotFoundException suppressed = new FileNotFoundException();
    ThrowableExtension.addSuppressed(receiver, suppressed);
    assertThat(ThrowableExtension.getSuppressed(receiver))
        .asList()
        .containsExactly((Object[]) receiver.getSuppressed());

    assertThat(printStackTracePrintStreamToString(receiver::printStackTrace))
        .isEqualTo(
            printStackTracePrintStreamToString(
                stream -> ThrowableExtension.printStackTrace(receiver, stream)));

    assertThat(printStackTracePrintWriterToString(receiver::printStackTrace))
        .isEqualTo(
            printStackTracePrintWriterToString(
                writer -> ThrowableExtension.printStackTrace(receiver, writer)));

    assertThat(printStackTraceStderrToString(receiver::printStackTrace))
        .isEqualTo(
            printStackTraceStderrToString(() -> ThrowableExtension.printStackTrace(receiver)));
  }

  /** This class */
  private static class ExceptionForTest extends Exception {

    private final MimicDesugaringStrategy strategy;

    public ExceptionForTest(MimicDesugaringStrategy strategy) {
      this.strategy = strategy;
    }

    @Override
    public void printStackTrace() {
      this.printStackTrace(System.err);
    }

    /**
     * This method should call this.printStackTrace(PrintWriter) directly. I deliberately change it
     * to strategy.printStackTrace(Throwable, PrintWriter) to simulate the behavior of Desguar, that
     * is, the direct call is intercepted and redirected to ThrowableExtension.
     */
    @Override
    public void printStackTrace(PrintStream s) {
      this.strategy.printStackTrace(
          this, new PrintWriter(new BufferedWriter(new OutputStreamWriter(s, UTF_8))));
    }
  }

  @Test
  public void testStrategySelection() throws ClassNotFoundException, IOException {
    String expectedStrategyClassName = getTwrStrategyClassNameSpecifiedInSystemProperty();
    assertThat(expectedStrategyClassName).isNotEmpty();
    assertThat(ThrowableExtension.STRATEGY.getClass().getName())
        .isEqualTo(expectedStrategyClassName);

    Class<?> expectedStrategyClass = Class.forName(expectedStrategyClassName);
    if (expectedStrategyClass.equals(ReuseDesugaringStrategy.class)) {
      testThrowableExtensionWithReuseDesugaringStrategy();
    } else if (expectedStrategyClass.equals(MimicDesugaringStrategy.class)) {
      testThrowableExtensionWithMimicDesugaringStrategy();
    } else if (expectedStrategyClass.equals(NullDesugaringStrategy.class)) {
      testThrowableExtensionWithNullDesugaringStrategy();
    } else {
      fail("unrecognized expected strategy class " + expectedStrategyClassName);
    }
  }

  private static int countOccurrences(String string, String substring) {
    int i = 0;
    int count = 0;
    while ((i = string.indexOf(substring, i)) >= 0) {
      ++count;
      i = i + string.length();
    }
    return count;
  }

  /** A mocked closeable class, which we can query the closedness. */
  private abstract static class AbstractResource {
    private final boolean exceptionOnClose;
    private boolean closed;

    protected AbstractResource(boolean exceptionOnClose) {
      this.exceptionOnClose = exceptionOnClose;
    }

    boolean isClosed() {
      return closed;
    }

    void internalClose() throws IOException {
      if (exceptionOnClose) {
        throw new IOException("intended exception");
      }
      closed = true;
    }
  }
}
