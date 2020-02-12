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
import com.google.devtools.build.lib.analysis.skylark.SkylarkModules; // a bad dependency
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.SyntaxError;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParsingException;
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

  private StarlarkSemantics semantics = StarlarkSemantics.DEFAULT_SEMANTICS;
  private final Map<String, Object> extraPredeclared = new HashMap<>();
  private StarlarkThread thread;

  @Before
  public final void initialize() {
    // TODO(adonovan): clean up the lazy initialization of thread when we disentangle
    // Module from it. Only the module need exist early; the thread can be created
    // immediately before execution
    thread = newStarlarkThread();
  }

  // Adds a binding to the predeclared environment.
  protected final void predeclare(String name, Object value) {
    extraPredeclared.put(name, value);
  }

  /**
   * Returns a new thread using the semantics set by setSemantics(), the predeclared environment of
   * SkylarkModules and prior calls to predeclared(), and a new mutability. Overridden by
   * subclasses.
   */
  public StarlarkThread newStarlarkThread() {
    ImmutableMap.Builder<String, Object> envBuilder = ImmutableMap.builder();
    SkylarkModules.addSkylarkGlobalsToBuilder(envBuilder); // TODO(adonovan): break bad dependency
    envBuilder.putAll(extraPredeclared);

    StarlarkThread thread =
        StarlarkThread.builder(Mutability.create("test"))
            .setGlobals(Module.createForBuiltins(envBuilder.build()))
            .setSemantics(semantics)
            .build();
    thread.setPrintHandler(StarlarkThread.makeDebugPrintHandler(getEventHandler()));
    return thread;
  }

  /**
   * Parses the semantics flags and updates the semantics used for subsequent evaluations. Also
   * reinitializes the thread.
   */
  protected final void setSemantics(String... options) throws OptionsParsingException {
    this.semantics =
        Options.parse(StarlarkSemanticsOptions.class, options).getOptions().toSkylarkSemantics();

    // Re-initialize the thread with the new semantics. See note at initialize.
    thread = newStarlarkThread();
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
    return EvalUtils.eval(input, thread.getGlobals(), thread);
  }

  /** Joins the lines, parses them as a file, and executes it. */
  public final void exec(String... lines) throws SyntaxError, EvalException, InterruptedException {
    ParserInput input = ParserInput.fromLines(lines);
    EvalUtils.exec(input, thread.getGlobals(), thread);
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

  /** Encapsulates a separate test which can be executed by a Scenario. */
  protected interface Testable {
    void run() throws Exception;
  }

  /**
   * A test scenario (a script of steps). Beware: Scenario is an inner class that mutates its
   * enclosing EvaluationTestCase as it executes the script.
   */
  public final class Scenario {
    private final SetupActions setup = new SetupActions();
    private final String[] skylarkOptions;

    public Scenario(String... skylarkOptions) {
      this.skylarkOptions = skylarkOptions;
    }

    private void run(Testable testable) throws Exception {
      setSemantics(skylarkOptions);
      testable.run();
    }

    /** Allows the execution of several statements before each following test. */
    public Scenario setUp(String... lines) {
      setup.registerExec(lines);
      return this;
    }

    /**
     * Allows the update of the specified variable before each following test
     *
     * @param name The name of the variable that should be updated
     * @param value The new value of the variable
     * @return This {@code Scenario}
     */
    public Scenario update(String name, Object value) {
      setup.registerUpdate(name, value);
      return this;
    }

    /**
     * Evaluates two expressions and asserts that their results are equal.
     *
     * @param src The source expression to be evaluated
     * @param expectedEvalString The expression of the expected result
     * @return This {@code Scenario}
     * @throws Exception
     */
    public Scenario testEval(String src, String expectedEvalString) throws Exception {
      runTest(createComparisonTestable(src, expectedEvalString, true));
      return this;
    }

    /** Evaluates an expression and compares its result to the expected object. */
    public Scenario testExpression(String src, Object expected) throws Exception {
      runTest(createComparisonTestable(src, expected, false));
      return this;
    }

    /** Evaluates an expression and compares its result to the ordered list of expected objects. */
    public Scenario testExactOrder(String src, Object... items) throws Exception {
      runTest(collectionTestable(src, items));
      return this;
    }

    /** Evaluates an expression and checks whether it fails with the expected error. */
    public Scenario testIfExactError(String expectedError, String... lines) throws Exception {
      runTest(errorTestable(true, expectedError, lines));
      return this;
    }

    /** Evaluates the expresson and checks whether it fails with the expected error. */
    public Scenario testIfErrorContains(String expectedError, String... lines) throws Exception {
      runTest(errorTestable(false, expectedError, lines));
      return this;
    }

    /** Looks up the value of the specified variable and compares it to the expected value. */
    public Scenario testLookup(String name, Object expected) throws Exception {
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
}
