// Copyright 2006 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax.util;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.SyntaxError;
import com.google.devtools.build.lib.syntax.ValidationEnvironment;
import com.google.devtools.build.lib.testutil.TestMode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import org.junit.Before;

/**
 * Base class for test cases that use parsing and evaluation services.
 */
public class EvaluationTestCase {
  private EventCollectionApparatus eventCollectionApparatus =
      new EventCollectionApparatus(EventKind.ALL_EVENTS);
  private TestMode testMode = TestMode.SKYLARK;
  protected StarlarkThread thread;
  protected Mutability mutability = Mutability.create("test");

  @Before
  public final void initialize() throws Exception {
    thread = newStarlarkThread();
  }

  /**
   * Creates a new StarlarkThread suitable for the test case. Subclasses may override it to fit
   * their purpose and e.g. call newBuildStarlarkThread or newStarlarkThread; or they may play with
   * the testMode to run tests in either or both kinds of StarlarkThread. Note that all
   * StarlarkThread-s may share the same Mutability, so don't close it.
   *
   * @return a fresh StarlarkThread.
   */
  public StarlarkThread newStarlarkThread() throws Exception {
    return newStarlarkThreadWithSkylarkOptions();
  }

  protected StarlarkThread newStarlarkThreadWithSkylarkOptions(String... skylarkOptions)
      throws Exception {
    return newStarlarkThreadWithBuiltinsAndSkylarkOptions(ImmutableMap.of(), skylarkOptions);
  }

  protected StarlarkThread newStarlarkThreadWithBuiltinsAndSkylarkOptions(
      Map<String, Object> builtins, String... skylarkOptions) throws Exception {
    if (testMode == null) {
      throw new IllegalArgumentException(
          "TestMode is null. Please set a Testmode via setMode() or set the "
              + "StarlarkThread manually by overriding newStarlarkThread()");
    }
    return testMode.createStarlarkThread(
        StarlarkThread.makeDebugPrintHandler(getEventHandler()), builtins, skylarkOptions);
  }

  /**
   * Sets the specified {@code TestMode} and tries to create the appropriate {@code StarlarkThread}
   *
   * @param testMode
   * @throws Exception
   */
  protected void setMode(TestMode testMode, String... skylarkOptions) throws Exception {
    this.testMode = testMode;
    thread = newStarlarkThreadWithSkylarkOptions(skylarkOptions);
  }

  protected void setMode(TestMode testMode, Map<String, Object> builtins,
      String... skylarkOptions) throws Exception {
    this.testMode = testMode;
    thread = newStarlarkThreadWithBuiltinsAndSkylarkOptions(builtins, skylarkOptions);
  }

  protected void enableSkylarkMode(Map<String, Object> builtins,
      String... skylarkOptions) throws Exception {
    setMode(TestMode.SKYLARK, builtins, skylarkOptions);
  }

  protected void enableSkylarkMode(String... skylarkOptions) throws Exception {
    setMode(TestMode.SKYLARK, skylarkOptions);
  }

  protected void enableBuildMode(String... skylarkOptions) throws Exception {
    setMode(TestMode.BUILD, skylarkOptions);
  }

  public ExtendedEventHandler getEventHandler() {
    return eventCollectionApparatus.reporter();
  }

  public StarlarkThread getStarlarkThread() {
    return thread;
  }

  // TODO(adonovan): don't let subclasses inherit vaguely specified "helpers".
  // Separate all the tests clearly into tests of the scanner, parser, resolver,
  // and evaluation.

  /** Parses an expression. */
  protected final Expression parseExpression(String... lines) throws SyntaxError {
    return Expression.parse(ParserInput.fromLines(lines));
  }

  /** Updates a binding in the module associated with the thread. */
  public EvaluationTestCase update(String varname, Object value) throws Exception {
    thread.getGlobals().put(varname, value);
    return this;
  }

  /** Returns the value of a binding in the module associated with the thread. */
  public Object lookup(String varname) throws Exception {
    return thread.getGlobals().lookup(varname);
  }

  /** Joins the lines, parses them as an expression, and evaluates it. */
  public final Object eval(String... lines) throws Exception {
    ParserInput input = ParserInput.fromLines(lines);
    return EvalUtils.eval(input, thread);
  }

  /** Joins the lines, parses them as a file, and executes it. */
  // TODO(adonovan): this function does too much:
  // - two modes, BUILD vs Skylark.
  // - parse + validate + BUILD dialect checks + execute.
  // Break the tests down into tests of just the scanner, parser, validator, build dialect checks,
  // or execution, and assert that all passes except the one of interest succeed.
  // All BUILD-dialect stuff belongs in bazel proper (lib.packages), not here.
  public final void exec(String... lines) throws Exception {
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input);
    ValidationEnvironment.validateFile(
        file, thread.getGlobals(), thread.getSemantics(), testMode == TestMode.BUILD);
    if (testMode == TestMode.SKYLARK) { // .bzl and other dialects
      if (!file.ok()) {
        throw new SyntaxError(file.errors());
      }
    } else {
      // For BUILD mode, validation events are reported but don't (yet)
      // prevent execution. We also apply BUILD dialect syntax checks.
      Event.replayEventsOn(getEventHandler(), file.errors());
      List<String> globs = new ArrayList<>(); // unused
      PackageFactory.checkBuildSyntax(file, globs, globs, new HashMap<>(), getEventHandler());
    }
    EvalUtils.exec(file, thread);
  }

  public void checkEvalError(String msg, String... input) throws Exception {
    try {
      exec(input);
      fail("Expected error '" + msg + "' but got no error");
    } catch (SyntaxError | EvalException | EventCollectionApparatus.FailFastException e) {
      assertThat(e).hasMessageThat().isEqualTo(msg);
    }
  }

  public void checkEvalErrorContains(String msg, String... input) throws Exception {
    try {
      exec(input);
      fail("Expected error containing '" + msg + "' but got no error");
    } catch (SyntaxError | EvalException | EventCollectionApparatus.FailFastException e) {
      assertThat(e).hasMessageThat().contains(msg);
    }
  }

  public void checkEvalErrorDoesNotContain(String msg, String... input) throws Exception {
    try {
      exec(input);
    } catch (SyntaxError | EvalException | EventCollectionApparatus.FailFastException e) {
      assertThat(e).hasMessageThat().doesNotContain(msg);
    }
  }

  // Forward relevant methods to the EventCollectionApparatus
  public EvaluationTestCase setFailFast(boolean failFast) {
    eventCollectionApparatus.setFailFast(failFast);
    return this;
  }

  public EvaluationTestCase assertNoWarningsOrErrors() {
    eventCollectionApparatus.assertNoWarningsOrErrors();
    return this;
  }

  public EventCollector getEventCollector() {
    return eventCollectionApparatus.collector();
  }

  public Event assertContainsError(String expectedMessage) {
    return eventCollectionApparatus.assertContainsError(expectedMessage);
  }

  public Event assertContainsWarning(String expectedMessage) {
    return eventCollectionApparatus.assertContainsWarning(expectedMessage);
  }

  public Event assertContainsDebug(String expectedMessage) {
    return eventCollectionApparatus.assertContainsDebug(expectedMessage);
  }

  public EvaluationTestCase clearEvents() {
    eventCollectionApparatus.clear();
    return this;
  }

  /**
   * Encapsulates a separate test which can be executed by a {@code TestMode}
   */
  protected interface Testable {
    public void run() throws Exception;
  }

  /**
   * Base class for test cases that run in specific modes (e.g. Build and/or Skylark)
   */
  protected abstract class ModalTestCase {
    private final SetupActions setup;

    protected ModalTestCase() {
      setup = new SetupActions();
    }

    /** Allows the execution of several statements before each following test. */
    public ModalTestCase setUp(String... lines) {
      setup.registerExec(lines);
      return this;
    }

    /**
     * Allows the update of the specified variable before each following test
     * @param name The name of the variable that should be updated
     * @param value The new value of the variable
     * @return This {@code ModalTestCase}
     */
    public ModalTestCase update(String name, Object value) {
      setup.registerUpdate(name, value);
      return this;
    }

    /**
     * Evaluates two expressions and asserts that their results are equal.
     *
     * @param src The source expression to be evaluated
     * @param expectedEvalString The expression of the expected result
     * @return This {@code ModalTestCase}
     * @throws Exception
     */
    public ModalTestCase testEval(String src, String expectedEvalString) throws Exception {
      runTest(createComparisonTestable(src, expectedEvalString, true));
      return this;
    }

    /** Evaluates an expression and compares its result to the expected object. */
    public ModalTestCase testExpression(String src, Object expected) throws Exception {
      runTest(createComparisonTestable(src, expected, false));
      return this;
    }

    /** Evaluates an expression and compares its result to the ordered list of expected objects. */
    public ModalTestCase testExactOrder(String src, Object... items) throws Exception {
      runTest(collectionTestable(src, items));
      return this;
    }

    /** Evaluates an expression and checks whether it fails with the expected error. */
    public ModalTestCase testIfExactError(String expectedError, String... lines) throws Exception {
      runTest(errorTestable(true, expectedError, lines));
      return this;
    }

    /** Evaluates the expresson and checks whether it fails with the expected error. */
    public ModalTestCase testIfErrorContains(String expectedError, String... lines)
        throws Exception {
      runTest(errorTestable(false, expectedError, lines));
      return this;
    }

    /** Looks up the value of the specified variable and compares it to the expected value. */
    public ModalTestCase testLookup(String name, Object expected) throws Exception {
      runTest(createLookUpTestable(name, expected));
      return this;
    }

    /**
     * Creates a Testable that checks whether the evaluation of the given expression fails with the
     * expected error.
     *
     * @param exactMatch whether the error message must be identical to the expected error.
     */
    protected Testable errorTestable(
        final boolean exactMatch, final String error, final String... lines) {
      return new Testable() {
        @Override
        public void run() throws Exception {
          if (exactMatch) {
            checkEvalError(error, lines);
          } else {
            checkEvalErrorContains(error, lines);
          }
        }
      };
    }

    /**
     * Creates a Testable that checks whether the value of the expression is a sequence containing
     * the expected elements.
     */
    protected Testable collectionTestable(final String src, final Object... expected) {
      return new Testable() {
        @Override
        public void run() throws Exception {
          assertThat((Iterable<?>) eval(src)).containsExactly(expected).inOrder();
        }
      };
    }

    /**
     * Creates a testable that compares the value of the expression to a specified result.
     *
     * @param src The expression to be evaluated
     * @param expected Either the expected object or an expression whose evaluation leads to the
     *     expected object
     * @param expectedIsExpression Signals whether {@code expected} is an object or an expression
     * @return An instance of Testable that runs the comparison
     */
    protected Testable createComparisonTestable(
        final String src, final Object expected, final boolean expectedIsExpression) {
      return new Testable() {
        @Override
        public void run() throws Exception {
          Object actual = eval(src);
          Object realExpected = expected;

          // We could also print the actual object and compare the string to the expected
          // expression, but then the order of elements would matter.
          if (expectedIsExpression) {
            realExpected = eval((String) expected);
          }

          assertThat(actual).isEqualTo(realExpected);
        }
      };
    }

    /**
     * Creates a Testable that looks up the given variable and compares its value to the expected
     * value
     * @param name
     * @param expected
     * @return An instance of Testable that does both lookup and comparison
     */
    protected Testable createLookUpTestable(final String name, final Object expected) {
      return new Testable() {
        @Override
        public void run() throws Exception {
          assertThat(lookup(name)).isEqualTo(expected);
        }
      };
    }

    /**
     * Executes the given Testable
     * @param testable
     * @throws Exception
     */
    protected void runTest(Testable testable) throws Exception {
      run(new TestableDecorator(setup, testable));
    }

    protected abstract void run(Testable testable) throws Exception;
  }

  /**
   * A simple decorator that allows the execution of setup actions before running a {@code Testable}
   */
  static class TestableDecorator implements Testable {
    private final SetupActions setup;
    private final Testable decorated;

    public TestableDecorator(SetupActions setup, Testable decorated) {
      this.setup = setup;
      this.decorated = decorated;
    }

    /**
     * Executes all stored actions and updates plus the actual {@code Testable}
     */
    @Override
    public void run() throws Exception {
      setup.executeAll();
      decorated.run();
    }
  }

  /**
   * A container for collection actions that should be executed before a test
   */
  class SetupActions {
    private List<Testable> setup;

    public SetupActions() {
      setup = new LinkedList<>();
    }

    /**
     * Registers an update to a module variable to be bound before a test
     *
     * @param name
     * @param value
     */
    public void registerUpdate(final String name, final Object value) {
      setup.add(
          new Testable() {
            @Override
            public void run() throws Exception {
              EvaluationTestCase.this.update(name, value);
            }
          });
    }

    /** Registers a sequence of statements for execution prior to a test. */
    public void registerExec(final String... lines) {
      setup.add(
          new Testable() {
            @Override
            public void run() throws Exception {
              EvaluationTestCase.this.exec(lines);
            }
          });
    }

    /**
     * Executes all stored actions and updates
     * @throws Exception
     */
    public void executeAll() throws Exception {
      for (Testable testable : setup) {
        testable.run();
      }
    }
  }

  /**
   * A class that executes each separate test in both modes (Build and Skylark)
   */
  protected class BothModesTest extends ModalTestCase {
    private final String[] skylarkOptions;

    public BothModesTest(String... skylarkOptions) {
      this.skylarkOptions = skylarkOptions;
    }

    /**
     * Executes the given Testable in both Build and Skylark mode
     */
    @Override
    protected void run(Testable testable) throws Exception {
      enableSkylarkMode(skylarkOptions);
      try {
        testable.run();
      } catch (Exception e) {
        throw new Exception("While in Skylark mode", e);
      }

      enableBuildMode(skylarkOptions);
      try {
        testable.run();
      } catch (Exception e) {
        throw new Exception("While in Build mode", e);
      }
    }
  }

  /**
   * A class that runs all tests in Build mode
   */
  protected class BuildTest extends ModalTestCase {
    private final String[] skylarkOptions;

    public BuildTest(String... skylarkOptions) {
      this.skylarkOptions = skylarkOptions;
    }

    @Override
    protected void run(Testable testable) throws Exception {
      enableBuildMode(skylarkOptions);
      testable.run();
    }
  }

  /**
   * A class that runs all tests in Skylark mode
   */
  protected class SkylarkTest extends ModalTestCase {
    private final String[] skylarkOptions;
    private final Map<String, Object> builtins;

    public SkylarkTest(String... skylarkOptions) {
      this(ImmutableMap.of(), skylarkOptions);
    }

    public SkylarkTest(Map<String, Object> builtins, String... skylarkOptions) {
      this.builtins = builtins;
      this.skylarkOptions = skylarkOptions;
    }

    @Override
    protected void run(Testable testable) throws Exception {
      enableSkylarkMode(builtins, skylarkOptions);
      testable.run();
    }
  }
}
