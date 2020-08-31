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
import com.google.devtools.build.lib.analysis.starlark.StarlarkModules; // a bad dependency
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FileOptions;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.SyntaxError;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.LinkedList;
import java.util.List;

/** Helper class for tests that evaluate Starlark code. */
// TODO(adonovan): stop extending this class. Prefer composition over inheritance.
// Rename it to EvaluationApparatus for consistency.
//
// TODO(adonovan): make predeclared env + semantics more like normal parameters.
// The main challenge is when are the Thread and Module created?
// They should have a consistent semantics, and the predeclared environment
// cannot be changed after the Module is created.
// Also, the fact that exec/eval/update/lookup can be used directly
// or through a Scenario complicates the question of when we commit to
// predeclared env + semantics.
// For the most part, the predeclared env doesn't vary across a suite,
// so it could be a constructor parameter.
//
// TODO(adonovan): this helper class might be somewhat handy for testing core Starlark, but its
// widespread use in tests of Bazel features greatly hinders the improvement of Bazel's loading
// phase. The existence of tests based on this class forces Bazel to continue support scenarios in
// which the test creates the environment, the threads, and so on, when these should be
// implemenation details of the loading phase. Instead, the lib.packages should present an API in
// which the client provides files, flags, and arguments like a command-line tool, and all our tests
// should be ported to use that API.
public class EvaluationTestCase {
  private EventCollectionApparatus eventCollectionApparatus =
      new EventCollectionApparatus(EventKind.ALL_EVENTS);

  private StarlarkSemantics semantics = StarlarkSemantics.DEFAULT;
  private StarlarkThread thread = null; // created lazily by getStarlarkThread
  private Module module = null; // created lazily by getModule

  /**
   * Parses the semantics flags and updates the semantics used to filter predeclared bindings, and
   * carried by subsequently created threads. Causes a new StarlarkThread and Module to be created
   * when next needed.
   */
  public final void setSemantics(String... options) throws OptionsParsingException {
    this.semantics =
        Options.parse(StarlarkSemanticsOptions.class, options).getOptions().toStarlarkSemantics();

    // Re-initialize the thread and module with the new semantics when needed.
    this.thread = null;
    this.module = null;
  }

  public ExtendedEventHandler getEventHandler() {
    return eventCollectionApparatus.reporter();
  }

  // TODO(adonovan): don't let subclasses inherit vaguely specified "helpers".
  // Separate all the tests clearly into tests of the scanner, parser, resolver,
  // and evaluation.

  /** Parses an expression. */
  protected final Expression parseExpression(String... lines) throws SyntaxError.Exception {
    return Expression.parse(ParserInput.fromLines(lines));
  }

  /** Updates a global binding in the module. */
  // TODO(adonovan): rename setGlobal.
  public EvaluationTestCase update(String varname, Object value) throws Exception {
    getModule().setGlobal(varname, value);
    return this;
  }

  /** Returns the value of a global binding in the module. */
  // TODO(adonovan): rename getGlobal.
  public Object lookup(String varname) throws Exception {
    return getModule().getGlobal(varname);
  }

  /** Joins the lines, parses them as an expression, and evaluates it. */
  public final Object eval(String... lines) throws Exception {
    ParserInput input = ParserInput.fromLines(lines);
    return Starlark.eval(input, FileOptions.DEFAULT, getModule(), getStarlarkThread());
  }

  /** Joins the lines, parses them as a file, and executes it. */
  public final void exec(String... lines)
      throws SyntaxError.Exception, EvalException, InterruptedException {
    ParserInput input = ParserInput.fromLines(lines);
    Starlark.execFile(input, FileOptions.DEFAULT, getModule(), getStarlarkThread());
  }

  // A hook for subclasses to alter a newly created thread,
  // e.g. by inserting thread-local values.
  protected void newThreadHook(StarlarkThread thread) {}

  // A hook for subclasses to alter the created module.
  // Implementations may add to the predeclared environment,
  // and return the module's client data value.
  protected Object newModuleHook(ImmutableMap.Builder<String, Object> predeclared) {
    StarlarkModules.addStarlarkGlobalsToBuilder(
        predeclared); // TODO(adonovan): break bad dependency
    return null; // no client data
  }

  public StarlarkThread getStarlarkThread() {
    if (this.thread == null) {
      Mutability mu = Mutability.create("test");
      StarlarkThread thread = new StarlarkThread(mu, semantics);
      thread.setPrintHandler(Event.makeDebugPrintHandler(getEventHandler()));
      newThreadHook(thread);
      this.thread = thread;
    }
    return this.thread;
  }

  public Module getModule() {
    if (this.module == null) {
      ImmutableMap.Builder<String, Object> predeclared = ImmutableMap.builder();
      Object clientData = newModuleHook(predeclared);
      Module module = Module.withPredeclared(semantics, predeclared.build());
      module.setClientData(clientData);
      this.module = module;
    }
    return this.module;
  }

  public void checkEvalError(String msg, String... input) throws Exception {
    try {
      exec(input);
      fail("Expected error '" + msg + "' but got no error");
    } catch (SyntaxError.Exception | EvalException | EventCollectionApparatus.FailFastException e) {
      assertThat(e).hasMessageThat().isEqualTo(msg);
    }
  }

  public void checkEvalErrorContains(String msg, String... input) throws Exception {
    try {
      exec(input);
      fail("Expected error containing '" + msg + "' but got no error");
    } catch (SyntaxError.Exception | EvalException | EventCollectionApparatus.FailFastException e) {
      assertThat(e).hasMessageThat().contains(msg);
    }
  }

  public void checkEvalErrorDoesNotContain(String msg, String... input) throws Exception {
    try {
      exec(input);
    } catch (SyntaxError.Exception | EvalException | EventCollectionApparatus.FailFastException e) {
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
    private final String[] starlarkOptions;

    public Scenario(String... starlarkOptions) {
      this.starlarkOptions = starlarkOptions;
    }

    private void run(Testable testable) throws Exception {
      setSemantics(starlarkOptions);
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
    private Testable errorTestable(
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
    private Testable collectionTestable(final String src, final Object... expected) {
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
    private Testable createComparisonTestable(
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
     *
     * @param name
     * @param expected
     * @return An instance of Testable that does both lookup and comparison
     */
    private Testable createLookUpTestable(final String name, final Object expected) {
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
