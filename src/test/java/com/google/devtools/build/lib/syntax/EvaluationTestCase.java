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

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.rules.SkylarkModules;

import org.junit.Before;

import java.util.List;

/**
 * Base class for test cases that use parsing and evaluation services.
 */
public class EvaluationTestCase {

  private EventCollectionApparatus eventCollectionApparatus;
  protected EvaluationContext evaluationContext;

  @Before
  public void setUp() throws Exception {
    eventCollectionApparatus = new EventCollectionApparatus(EventKind.ALL_EVENTS);
    evaluationContext = newEvaluationContext();
  }

  protected EventHandler getEventHandler() {
    return eventCollectionApparatus.reporter();
  }

  public Environment getEnvironment() {
    return evaluationContext.getEnvironment();
  }

  public EvaluationContext newEvaluationContext() throws Exception {
    return SkylarkModules.newEvaluationContext(getEventHandler());
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
}
