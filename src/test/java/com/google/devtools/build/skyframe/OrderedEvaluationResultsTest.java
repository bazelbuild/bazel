// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests ordered delivery and keyed fallback across memoizing evaluation waves. */
@RunWith(JUnit4.class)
public final class OrderedEvaluationResultsTest {
  private static final SkyFunctionName FUNCTION =
      SkyFunctionName.createHermetic("ORDERED_EVALUATION_RESULTS_TEST");

  private record Key(String name) implements SkyKey {
    @Override
    public SkyFunctionName functionName() {
      return FUNCTION;
    }
  }

  private record Value(String text) implements SkyValue {}

  private static final class Failure extends SkyFunctionException {
    Failure() {
      super(new IOException("failed root"), Transience.PERSISTENT);
    }
  }

  private static EvaluationContext context(boolean ordered, boolean keepGoing) {
    return EvaluationContext.newBuilder()
        .setEventHandler(ExtendedEventHandler.NOOP)
        .setParallelism(2)
        .setKeepGoing(keepGoing)
        .setOrderedRootResults(ordered)
        .build();
  }

  private static InMemoryMemoizingEvaluator evaluator(SkyFunction function) {
    return new InMemoryMemoizingEvaluator(
        Map.of(FUNCTION, function), new SequencedRecordingDifferencer());
  }

  @Test
  public void mixedAndCachedResultsUsePositionsOnlyForOriginalRoots() throws Exception {
    var cached = new Key("cached");
    var fresh = new Key("fresh");
    var failed = new Key("failed");
    var evaluator =
        evaluator(
            (key, env) -> {
              if (key.equals(failed)) {
                throw new Failure();
              }
              return new Value(((Key) key).name());
            });
    evaluator.evaluate(ImmutableSet.of(cached), context(false, true));

    ImmutableSet<SkyKey> roots = ImmutableSet.of(fresh, failed, cached);
    EvaluationContext ordered = context(true, true).builder().build();
    for (int run = 0; run < 2; run++) {
      var result = evaluator.evaluate(roots, ordered);
      var positions = result.orderedValuesFor(roots);
      assertNotNull(positions);
      assertEquals(new Value("fresh"), positions.apply(0));
      assertNull(positions.apply(1));
      assertEquals(new Value("cached"), positions.apply(2));
      assertEquals("failed root", result.getError(failed).getException().getMessage());
      assertNull(result.orderedValuesFor(ImmutableSet.of(fresh, failed, cached)));
      assertNull(result.orderedValuesFor(ImmutableSet.of(cached, failed, fresh)));
      assertEquals(new Value("fresh"), result.get(fresh));
      assertEquals(new Value("cached"), result.get(cached));
      assertNull(result.get(failed));
    }

    var keyed = evaluator.evaluate(roots, context(false, true));
    assertNull(keyed.orderedValuesFor(roots));
    assertEquals(new Value("fresh"), keyed.get(fresh));
    assertEquals("failed root", keyed.getError(failed).getException().getMessage());
  }

  @Test
  public void cachedFailFastUsesKeyedResult() throws Exception {
    var failed = new Key("failed");
    var fresh = new Key("fresh");
    var evaluator =
        evaluator(
            (key, env) -> {
              throw new Failure();
            });
    evaluator.evaluate(ImmutableSet.of(failed), context(true, true));

    ImmutableSet<SkyKey> roots = ImmutableSet.of(failed, fresh);
    var result = evaluator.evaluate(roots, context(true, false));
    assertNull(result.orderedValuesFor(roots));
    assertEquals("failed root", result.getError(failed).getException().getMessage());
    assertNull(result.get(fresh));
    assertNull(result.getError(fresh));
  }

  @Test
  public void cycleUsesKeyedResult() throws Exception {
    var cached = new Key("cached");
    var cycle = new Key("cycle");
    var evaluator =
        evaluator((key, env) -> key.equals(cached) ? new Value("cached") : env.getValue(cycle));
    evaluator.evaluate(ImmutableSet.of(cached), context(true, true));

    ImmutableSet<SkyKey> roots = ImmutableSet.of(cached, cycle);
    var result = evaluator.evaluate(roots, context(true, true));
    assertNull(result.orderedValuesFor(roots));
    assertEquals(new Value("cached"), result.get(cached));
    assertFalse(result.getError(cycle).getCycleInfo().isEmpty());
  }
}
