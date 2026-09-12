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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WorkerSkyFunctionEnvironment}. */
@RunWith(JUnit4.class)
public final class WorkerSkyFunctionEnvironmentTest {
  private static final SkyKey KEY_A = GraphTester.skyKey("a");
  private static final SkyKey KEY_B = GraphTester.skyKey("b");
  private static final StringValue VALUE_A = new StringValue("a");
  private static final StringValue VALUE_B = new StringValue("b");

  private final AtomicInteger freshEnvRequests = new AtomicInteger();

  /**
   * A delegate that serves values, exceptions or missing deps from a fixed map, with the same
   * sticky {@link #valuesMissing} semantics as the real environment.
   */
  private static final class FakeEnvironment extends AbstractSkyFunctionEnvironmentForTesting {
    private final ImmutableMap<SkyKey, ValueOrUntypedException> deps;

    FakeEnvironment(ImmutableMap<SkyKey, ValueOrUntypedException> deps) {
      this.deps = deps;
    }

    private ValueOrUntypedException lookup(SkyKey key) {
      return deps.getOrDefault(key, ValueOrUntypedException.ofNull());
    }

    @Override
    protected ImmutableMap<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
        Iterable<? extends SkyKey> depKeys) {
      ImmutableMap.Builder<SkyKey, ValueOrUntypedException> result = ImmutableMap.builder();
      for (SkyKey key : depKeys) {
        ValueOrUntypedException voe = lookup(key);
        if (voe.getValue() == null && voe.getException() == null) {
          valuesMissing = true;
        }
        result.put(key, voe);
      }
      return result.buildKeepingLast();
    }

    @Override
    public SkyframeLookupResult getLookupHandleForPreviouslyRequestedDeps() {
      return new SimpleSkyframeLookupResult(() -> valuesMissing = true, this::lookup);
    }

    @Override
    public void dependOnFuture(ListenableFuture<?> future) {
      if (externalDeps == null) {
        externalDeps = new ArrayList<>();
      }
      externalDeps.add(future);
    }

    @Override
    public ExtendedEventHandler getListener() {
      throw new UnsupportedOperationException();
    }
  }

  private WorkerSkyFunctionEnvironment createWorkerEnv(
      FakeEnvironment initial, FakeEnvironment... fresh) {
    Iterator<FakeEnvironment> freshEnvs = ImmutableList.copyOf(fresh).iterator();
    InterruptibleSupplier<SkyFunction.Environment> supplier =
        () -> {
          freshEnvRequests.incrementAndGet();
          assertThat(freshEnvs.hasNext()).isTrue();
          return freshEnvs.next();
        };
    return new WorkerSkyFunctionEnvironment(initial, supplier);
  }

  @Test
  public void missingDep_blocksForFreshEnvironmentAndRetries() throws Exception {
    FakeEnvironment initial = new FakeEnvironment(ImmutableMap.of());
    FakeEnvironment fresh =
        new FakeEnvironment(
            ImmutableMap.of(KEY_A, ValueOrUntypedException.ofValueUntyped(VALUE_A)));
    WorkerSkyFunctionEnvironment workerEnv = createWorkerEnv(initial, fresh);

    assertThat(workerEnv.getValue(KEY_A)).isEqualTo(VALUE_A);
    assertThat(freshEnvRequests.get()).isEqualTo(1);
  }

  @Test
  public void earlierUnhandledError_doesNotForceFreshEnvironmentForAvailableDep()
      throws Exception {
    FakeEnvironment env =
        new FakeEnvironment(
            ImmutableMap.of(
                KEY_A, ValueOrUntypedException.ofExn(new SomeErrorException("a")),
                KEY_B, ValueOrUntypedException.ofValueUntyped(VALUE_B)));
    WorkerSkyFunctionEnvironment workerEnv = createWorkerEnv(env);

    // The dep is in error, but not with a handled exception type, so the lookup yields null and the
    // delegate's valuesMissing flag becomes sticky.
    assertThat(workerEnv.getValueOrThrow(KEY_A, IOException.class)).isNull();
    assertThat(env.valuesMissing()).isTrue();

    assertThat(workerEnv.getValue(KEY_B)).isEqualTo(VALUE_B);
    assertThat(
            workerEnv.getValueOrThrow(
                KEY_B,
                IOException.class,
                java.util.concurrent.TimeoutException.class,
                ReflectiveOperationException.class,
                CloneNotSupportedException.class))
        .isEqualTo(VALUE_B);
    assertThat(freshEnvRequests.get()).isEqualTo(0);
  }

  @Test
  public void unhandledError_returnsNullWithoutFreshEnvironment() throws Exception {
    FakeEnvironment env =
        new FakeEnvironment(
            ImmutableMap.of(KEY_A, ValueOrUntypedException.ofExn(new SomeErrorException("a"))));
    WorkerSkyFunctionEnvironment workerEnv = createWorkerEnv(env);

    assertThat(workerEnv.getValueOrThrow(KEY_A, IOException.class)).isNull();
    assertThat(
            workerEnv.getValueOrThrow(
                KEY_A,
                IOException.class,
                java.util.concurrent.TimeoutException.class,
                ReflectiveOperationException.class,
                CloneNotSupportedException.class))
        .isNull();
    assertThat(freshEnvRequests.get()).isEqualTo(0);
  }

  @Test
  public void handledError_isThrownWithoutFreshEnvironment() {
    FakeEnvironment env =
        new FakeEnvironment(
            ImmutableMap.of(KEY_A, ValueOrUntypedException.ofExn(new SomeErrorException("a"))));
    WorkerSkyFunctionEnvironment workerEnv = createWorkerEnv(env);

    assertThrows(
        SomeErrorException.class, () -> workerEnv.getValueOrThrow(KEY_A, SomeErrorException.class));
    assertThat(freshEnvRequests.get()).isEqualTo(0);
  }

  @Test
  public void registeredFuture_doesNotForceFreshEnvironmentForAvailableDep() throws Exception {
    FakeEnvironment env =
        new FakeEnvironment(
            ImmutableMap.of(KEY_B, ValueOrUntypedException.ofValueUntyped(VALUE_B)));
    WorkerSkyFunctionEnvironment workerEnv = createWorkerEnv(env);

    workerEnv.dependOnFuture(SettableFuture.create());
    assertThat(env.valuesMissing()).isTrue();

    assertThat(workerEnv.getValue(KEY_B)).isEqualTo(VALUE_B);
    assertThat(freshEnvRequests.get()).isEqualTo(0);
  }

  @Test
  public void batchLookup_blocksOnlyIfSomeDepIsMissing() throws Exception {
    FakeEnvironment initial =
        new FakeEnvironment(
            ImmutableMap.of(KEY_A, ValueOrUntypedException.ofValueUntyped(VALUE_A)));
    FakeEnvironment fresh =
        new FakeEnvironment(
            ImmutableMap.of(
                KEY_A, ValueOrUntypedException.ofValueUntyped(VALUE_A),
                KEY_B, ValueOrUntypedException.ofValueUntyped(VALUE_B)));
    WorkerSkyFunctionEnvironment workerEnv = createWorkerEnv(initial, fresh);

    SkyframeLookupResult result = workerEnv.getValuesAndExceptions(ImmutableList.of(KEY_A, KEY_B));

    assertThat(freshEnvRequests.get()).isEqualTo(1);
    assertThat(result.get(KEY_A)).isEqualTo(VALUE_A);
    assertThat(result.get(KEY_B)).isEqualTo(VALUE_B);
  }
}
