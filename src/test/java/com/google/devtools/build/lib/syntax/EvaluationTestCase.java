// Copyright 2006-2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.truth.Ordered;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.testutil.TestMode;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;

import org.junit.Before;

import java.util.LinkedList;
import java.util.List;

/**
 * Base class for test cases that use parsing and evaluation services.
 */
public class EvaluationTestCase {
  
  private EventCollectionApparatus eventCollectionApparatus;
  private PackageFactory factory;
  private TestMode testMode = TestMode.SKYLARK;
  
  protected EvaluationContext evaluationContext;

  public EvaluationTestCase()   {
    createNewInfrastructure();
  }
  
  @Before
  public void setUp() throws Exception {
    createNewInfrastructure();
    evaluationContext = newEvaluationContext();
  }

  public EvaluationContext newEvaluationContext() throws Exception {
    if (testMode == null) {
      throw new IllegalArgumentException(
          "TestMode is null. Please set a Testmode via setMode() or set the "
          + "evaluatenContext manually by overriding newEvaluationContext()");
    }

    return testMode.createContext(getEventHandler(), factory.getEnvironment());
  }

  protected void createNewInfrastructure()  {
    eventCollectionApparatus = new EventCollectionApparatus(EventKind.ALL_EVENTS);
    factory = new PackageFactory(TestRuleClassProvider.getRuleClassProvider());
  }
  
  /**
   * Sets the specified {@code TestMode} and tries to create the appropriate {@code
   * EvaluationContext}
   * 
   * @param testMode
   * @throws Exception
   */
  protected void setMode(TestMode testMode) throws Exception {
    this.testMode = testMode;
    evaluationContext = newEvaluationContext();
  }

  protected void enableSkylarkMode() throws Exception {
    setMode(TestMode.SKYLARK);
  }

  protected void enableBuildMode() throws Exception {
    setMode(TestMode.BUILD);
  }

  protected EventHandler getEventHandler() {
    return eventCollectionApparatus.reporter();
  }

  protected PackageFactory getFactory() {
    return factory;
  }

  public Environment getEnvironment() {
    return evaluationContext.getEnvironment();
  }
  
  public boolean isSkylark() {
    return evaluationContext.isSkylark();
  }

  protected List<Statement> parseFile(String... input) {
    return evaluationContext.parseFile(input);
  }

  Expression parseExpression(String... input) {
    return evaluationContext.parseExpression(input);
  }

  public EvaluationTestCase update(String varname, Object value) throws Exception {
    evaluationContext.update(varname, value);
    return this;
  }

  public Object lookup(String varname) throws Exception {
    return evaluationContext.lookup(varname);
  }

  public Object eval(String... input) throws Exception {
    return evaluationContext.eval(input);
  }

  public void checkEvalError(String msg, String... input) throws Exception {
    setFailFast(true);
    try {
      eval(input);
      fail();
    } catch (IllegalArgumentException | EvalException e) {
      assertThat(e).hasMessage(msg);
    }
  }

  public void checkEvalErrorContains(String msg, String... input) throws Exception {
    try {
      eval(input);
      fail();
    } catch (IllegalArgumentException | EvalException e) {
      assertThat(e.getMessage()).contains(msg);
    }
  }

  public void checkEvalErrorStartsWith(String msg, String... input) throws Exception {
    try {
      eval(input);
      fail();
    } catch (IllegalArgumentException | EvalException e) {
      assertThat(e.getMessage()).startsWith(msg);
    }
  }

  // Forward relevant methods to the EventCollectionApparatus
  public EvaluationTestCase setFailFast(boolean failFast) {
    eventCollectionApparatus.setFailFast(failFast);
    return this;
  }
  public EvaluationTestCase assertNoEvents() {
    eventCollectionApparatus.assertNoEvents();
    return this;
  }
  public EventCollector getEventCollector() {
    return eventCollectionApparatus.collector();
  }
  public Event assertContainsEvent(String expectedMessage) {
    return eventCollectionApparatus.assertContainsEvent(expectedMessage);
  }
  public List<Event> assertContainsEventWithFrequency(String expectedMessage,
      int expectedFrequency) {
    return eventCollectionApparatus.assertContainsEventWithFrequency(
        expectedMessage, expectedFrequency);
  }
  public Event assertContainsEventWithWordsInQuotes(String... words) {
    return eventCollectionApparatus.assertContainsEventWithWordsInQuotes(words);
  }
  public EvaluationTestCase clearEvents() {
    eventCollectionApparatus.collector().clear();
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
   *
   */  
  protected abstract class ModalTestCase {
    private final SetupActions setup;
    
    protected ModalTestCase()   {
      setup = new SetupActions();
    }
    
    /**
     * Allows the execution of several statements before each following test
     * @param statements The statement(s) to be executed
     * @return This {@code ModalTestCase}
     */
    public ModalTestCase setUp(String... statements) {
      setup.registerEval(statements);
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
     * Evaluates two parameters and compares their results.
     * @param statement The statement to be evaluated
     * @param expectedEvalString The expression of the expected result
     * @return This {@code ModalTestCase}
     * @throws Exception
     */
    public ModalTestCase testEval(String statement, String expectedEvalString) throws Exception {
      runTest(createComparisonTestable(statement, expectedEvalString, true));
      return this;
    }

    /**
     * Evaluates the given statement and compares its result to the expected object
     * @param statement
     * @param expected
     * @return This {@code ModalTestCase}
     * @throws Exception
     */
    public ModalTestCase testStatement(String statement, Object expected) throws Exception {
      runTest(createComparisonTestable(statement, expected, false));
      return this;
    }

    /**
     * Evaluates the given statement and compares its result to the collection of expected objects
     * without considering their order
     * @param statement The statement to be evaluated
     * @param items The expected items
     * @return This {@code ModalTestCase}
     * @throws Exception
     */
    public ModalTestCase testCollection(String statement, Object... items) throws Exception {
      runTest(collectionTestable(statement, false, items));
      return this;
    }
    
    /**
     * Evaluates the given statement and compares its result to the collection of expected objects
     * while considering their order
     * @param statement The statement to be evaluated
     * @param items The expected items, in order
     * @return This {@code ModalTestCase}
     * @throws Exception
     */
    public ModalTestCase testExactOrder(String statement, Object... items) throws Exception {
      runTest(collectionTestable(statement, true, items));
      return this;
    }

    /**
     * Evaluates the given statement and checks whether the given error message appears
     * @param expectedError The expected error message
     * @param statements The statement(s) to be evaluated
     * @return This ModalTestCase
     * @throws Exception
     */
    public ModalTestCase testIfExactError(String expectedError, String... statements)
        throws Exception {
      runTest(errorTestable(true, expectedError, statements));
      return this;
    }

    /**
     * Evaluates the given statement and checks whether an error that contains the expected message
     * occurs
     * @param expectedError
     * @param statements
     * @return This ModalTestCase
     * @throws Exception
     */
    public ModalTestCase testIfErrorContains(String expectedError, String... statements)
        throws Exception {
      runTest(errorTestable(false, expectedError, statements));
      return this;
    }

    /**
     * Looks up the value of the specified variable and compares it to the expected value
     * @param name
     * @param expected
     * @return This ModalTestCase
     * @throws Exception
     */
    public ModalTestCase testLookup(String name, Object expected) throws Exception {
      runTest(createLookUpTestable(name, expected));
      return this;
    }

    /**
     * Creates a Testable that checks whether the evaluation of the given statement leads to the
     * expected error
     * @param statements
     * @param error
     * @param exactMatch If true, the error message has to be identical to the expected error
     * @return An instance of Testable that runs the error check
     */
    protected Testable errorTestable(final boolean exactMatch, final String error,
        final String... statements) {
      return new Testable() {
        @Override
        public void run() throws Exception {
          if (exactMatch) {
            checkEvalError(error, statements);
          } else {
            checkEvalErrorContains(error, statements);
          }
        }
      };
    }

    /**
     * Creates a testable that checks whether the evaluation of the given statement leads to a list
     * that contains exactly the expected objects
     * @param statement The statement to be evaluated
     * @param ordered Determines whether the order of the elements is checked as well
     * @param expected Expected objects
     * @return An instance of Testable that runs the check
     */
    protected Testable collectionTestable(
        final String statement, final boolean ordered, final Object... expected) {
      return new Testable() {
        @Override
        public void run() throws Exception {
          Ordered tmp = assertThat((Iterable<?>) eval(statement)).containsExactly(expected);

          if (ordered) {
            tmp.inOrder();
          }
        }
      };
    }

    /**
     * Creates a testable that compares the evaluation of the given statement to a specified result
     *
     * @param statement The statement to be evaluated
     * @param expected Either the expected object or an expression whose evaluation leads to the
     *  expected object
     * @param expectedIsExpression Signals whether {@code expected} is an object or an expression
     * @return An instance of Testable that runs the comparison
     */
    protected Testable createComparisonTestable(
        final String statement, final Object expected, final boolean expectedIsExpression) {
      return new Testable() {
        @Override
        public void run() throws Exception {
          Object actual = eval(statement);
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
   * A simple decorator that allows the execution of setup actions before running 
   * a {@code Testable}
   */
  class TestableDecorator implements Testable {
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
     * Registers a variable that has to be updated before a test
     *
     * @param name
     * @param value
     */
    public void registerUpdate(final String name, final Object value) {
      setup.add(new Testable() {
        @Override
        public void run() throws Exception {
          EvaluationTestCase.this.update(name, value);
        }
      });
    }

    /**
     * Registers a statement for evaluation prior to a test
     *
     * @param statements
     */
    public void registerEval(final String... statements) {
      setup.add(new Testable() {
        @Override
        public void run() throws Exception {
          EvaluationTestCase.this.eval(statements);
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
    public BothModesTest() {}

    /**
     * Executes the given Testable in both Build and Skylark mode
     */
    @Override
    protected void run(Testable testable) throws Exception {
      enableSkylarkMode();
      try {
        testable.run();
      } catch (Exception e) {
        throw new Exception("While in Skylark mode", e);
      }

      enableBuildMode();
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
    public BuildTest() {}

    @Override
    protected void run(Testable testable) throws Exception {
      enableBuildMode();
      testable.run();
    }
  }

  /**
   * A class that runs all tests in Skylark mode
   */
  protected class SkylarkTest extends ModalTestCase {
    public SkylarkTest() {}

    @Override
    protected void run(Testable testable) throws Exception {
      enableSkylarkMode();
      testable.run();
    }
  }
}
