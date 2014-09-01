// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.GraphTester.COPY;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.skyframe.GraphTester.StringValue;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.concurrent.atomic.AtomicInteger;

import javax.annotation.Nullable;

/**
 * Tests for {@link TrivialVersionedEvaluator}.
 */
@RunWith(JUnit4.class)
public class TrivialVersionedEvaluatorTest {

  private VersionedEvaluatorTester tester;
  private EventCollector eventCollector;
  private EventHandler reporter;
  private MemoizingEvaluator.EmittedEventState emittedEventState;
  private Version curVersion;

  @Before
  public void initializeTester() {
    initializeTester(null);
  }

  public void initializeTester(@Nullable TrackingInvalidationReceiver customInvalidationReceiver) {
    emittedEventState = new MemoizingEvaluator.EmittedEventState();
    tester = new VersionedEvaluatorTester();
    if (customInvalidationReceiver != null) {
      tester.setInvalidationReceiver(customInvalidationReceiver);
    }
    tester.initialize();
  }

  @Before
  public void initializeReporter() {
    eventCollector = new EventCollector(EventKind.ALL_EVENTS);
    reporter = new Reporter(eventCollector);
    tester.resetPlayedEvents();
  }

  @Before
  public void initializeVersion() {
    curVersion = new IntVersion(23);
  }

  @Test
  public void smoke() throws Exception {
    tester.set("x", new StringValue("y"));
    StringValue value = (StringValue) tester.evalAndGet("x");
    assertEquals("y", value.getValue());
  }

  @Test
  public void simpleDependency() throws Exception {
    tester.getOrCreate("ab")
        .addDependency("a")
        .setComputedValue(COPY);
    tester.set("a", new StringValue("me"));
    StringValue value = (StringValue) tester.evalAndGet("ab");
    assertEquals("me", value.getValue());
  }

  @Test
  public void versionsAreDisjoint() throws Exception {
    final AtomicInteger numCalls = new AtomicInteger();
    tester.getOrCreate("v").setBuilder(new SkyFunction() {
      @Nullable
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) {
        IntVersion intVersion = (IntVersion) curVersion;
        numCalls.incrementAndGet();
        return new StringValue("i am at v" + intVersion.getVal());
      }

      @Nullable
      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    });

    StringValue value = (StringValue) tester.evalAndGet("v");
    assertEquals("i am at v23", value.getValue());

    value = (StringValue) tester.evalAndGet("v");
    assertEquals("i am at v23", value.getValue());

    curVersion = new IntVersion(123);
    value = (StringValue) tester.evalAndGet("v");
    assertEquals("i am at v123", value.getValue());

    // There should be 1 call for each version computed, not for each eval().
    assertEquals(2, numCalls.get());
  }

  /**
   * A graph tester that is specific to the TrivialVersionedEvaluator, with some convenience
   * methods.
   */
  private class VersionedEvaluatorTester extends GraphTester {
    private MemoizingEvaluator graph;
    private TrackingInvalidationReceiver invalidationReceiver = new TrackingInvalidationReceiver();

    public void initialize() {
      this.graph = new TrivialVersionedEvaluator(
          ImmutableMap.of(NODE_TYPE, createDelegatingFunction()),
          invalidationReceiver, emittedEventState, true);
    }

    public void setInvalidationReceiver(TrackingInvalidationReceiver customInvalidationReceiver) {
      Preconditions.checkState(graph == null, "graph already initialized");
      invalidationReceiver = customInvalidationReceiver;
    }

    public void resetPlayedEvents() {
      emittedEventState.clear();
    }

    public <T extends SkyValue> EvaluationResult<T> eval(
        boolean keepGoing, int numThreads, SkyKey... keys)
        throws InterruptedException {
      assertThat(getModifiedValues()).isEmpty();
      return graph.evaluate(ImmutableList.copyOf(keys), curVersion, keepGoing, numThreads,
          reporter);
    }

    public <T extends SkyValue> EvaluationResult<T> eval(boolean keepGoing, SkyKey... keys)
        throws InterruptedException {
      return eval(keepGoing, 100, keys);
    }

    public <T extends SkyValue> EvaluationResult<T> eval(boolean keepGoing,
                                                         String... keys)
        throws InterruptedException {
      return eval(keepGoing, toSkyKeys(keys));
    }

    public SkyValue evalAndGet(boolean keepGoing, String key)
        throws InterruptedException {
      return evalAndGet(keepGoing, new SkyKey(NODE_TYPE, key));
    }

    public SkyValue evalAndGet(String key) throws InterruptedException {
      return evalAndGet(/*keepGoing=*/false, key);
    }

    public SkyValue evalAndGet(boolean keepGoing, SkyKey key)
        throws InterruptedException {
      EvaluationResult<StringValue> evaluationResult = eval(keepGoing, key);
      SkyValue result = evaluationResult.get(key);
      assertNotNull(evaluationResult.toString(), result);
      return result;
    }
  }
}
