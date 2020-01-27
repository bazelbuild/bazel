// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.supplier;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link EvictableSupplier}. */
@RunWith(JUnit4.class)
public final class EvictableSupplierTest {

  @Test
  public void usesInitialCachedValueIfStillInMemory() throws Exception {
    Object initialCachedValue = new Object();
    EvictableSupplier<?> supplier =
        new EvictableSupplier<Object>(initialCachedValue) {
          @Override
          protected Object computeValue() {
            throw new AssertionError("Should not be called");
          }
        };

    Object result = supplier.get();

    assertThat(result).isSameInstanceAs(initialCachedValue);
  }

  @Test
  public void computesValue() throws Exception {
    Object computedValue = new Object();
    EvictableSupplier<?> supplier =
        new EvictableSupplier<Object>(/*cachedValue=*/ null) {
          @Override
          protected Object computeValue() {
            return computedValue;
          }
        };

    Object result = supplier.get();

    assertThat(result).isSameInstanceAs(computedValue);
  }

  @Test
  public void reusesComputedValueIfStillInMemory() throws Exception {
    EvictableSupplier<?> supplier =
        new EvictableSupplier<Object>(/*cachedValue=*/ null) {
          @Override
          protected Object computeValue() {
            return new Object();
          }
        };

    Object result1 = supplier.get();
    Object result2 = supplier.get();

    assertThat(result2).isSameInstanceAs(result1);
  }

  @Test
  public void onlyCallsComputeOnceIfResultStillInMemory() throws Exception {
    AtomicInteger callCount = new AtomicInteger(0);
    EvictableSupplier<?> supplier =
        new EvictableSupplier<Object>(/*cachedValue=*/ null) {
          @Override
          protected Object computeValue() {
            callCount.incrementAndGet();
            return new Object();
          }
        };

    @SuppressWarnings("unused") // Holding a strong reference.
    Object result = supplier.get();
    supplier.get();

    assertThat(callCount.get()).isEqualTo(1);
  }

  @Test
  public void canPeekAtInitialCachedValue() {
    Object initialCachedValue = new Object();
    EvictableSupplier<?> supplier =
        new EvictableSupplier<Object>(initialCachedValue) {
          @Override
          protected Object computeValue() {
            throw new AssertionError("Should not be called");
          }
        };

    Object cachedValue = supplier.peekCachedValue();

    assertThat(cachedValue).isSameInstanceAs(initialCachedValue);
  }

  @Test
  public void canPeekAtComputedValue() throws Exception {
    EvictableSupplier<?> supplier =
        new EvictableSupplier<Object>(/*cachedValue=*/ null) {
          @Override
          protected Object computeValue() {
            return new Object();
          }
        };

    Object result = supplier.get();
    Object cachedValue = supplier.peekCachedValue();

    assertThat(cachedValue).isSameInstanceAs(result);
  }

  @Test
  public void peekReturnsNullWhenValueNotComputed() {
    EvictableSupplier<?> supplier =
        new EvictableSupplier<Object>(/*cachedValue=*/ null) {
          @Override
          protected Object computeValue() {
            throw new AssertionError("Should not be called");
          }
        };

    Object cachedValue = supplier.peekCachedValue();

    assertThat(cachedValue).isNull();
  }

  @Test
  public void peekReturnsNullWhenInitialCachedValueEvicted() {
    EvictableSupplier<?> supplier =
        new EvictableSupplier<Object>(new Object()) {
          @Override
          protected Object computeValue() {
            throw new AssertionError("Should not be called");
          }
        };

    supplier.evictForTesting();
    Object cachedValue = supplier.peekCachedValue();

    assertThat(cachedValue).isNull();
  }

  @Test
  public void peekReturnsNullWhenComputedValueEvicted() throws Exception {
    EvictableSupplier<?> supplier =
        new EvictableSupplier<Object>(/*cachedValue=*/ null) {
          @Override
          protected Object computeValue() {
            return new Object();
          }
        };

    supplier.get();
    supplier.evictForTesting();
    Object cachedValue = supplier.peekCachedValue();

    assertThat(cachedValue).isNull();
  }

  @Test
  public void recomputesAfterEvictionOfInitialCachedValue() throws Exception {
    Object initialCachedValue = new Object();
    Object computedValue = new Object();
    EvictableSupplier<?> supplier =
        new EvictableSupplier<Object>(initialCachedValue) {
          @Override
          protected Object computeValue() {
            return computedValue;
          }
        };

    supplier.evictForTesting();
    Object result = supplier.get();

    assertThat(result).isSameInstanceAs(computedValue);
  }

  @Test
  public void recomputesAfterEvictionOfComputedValue() throws Exception {
    Object computedValue1 = new Object();
    Object computedValue2 = new Object();
    EvictableSupplier<?> supplier =
        new EvictableSupplier<Object>(/*cachedValue=*/ null) {
          int calls = 0;

          @Override
          protected Object computeValue() {
            ++calls;
            if (calls == 1) {
              return computedValue1;
            }
            if (calls == 2) {
              return computedValue2;
            }
            throw new AssertionError("Called " + calls + " times");
          }
        };

    Object result1 = supplier.get();
    supplier.evictForTesting();
    Object result2 = supplier.get();

    assertThat(result1).isSameInstanceAs(computedValue1);
    assertThat(result2).isSameInstanceAs(computedValue2);
  }

  @Test
  public void computeValueCannotReturnNull() {
    EvictableSupplier<?> supplier =
        new EvictableSupplier<Object>(/*cachedValue=*/ null) {
          @Override
          protected Object computeValue() {
            return null;
          }
        };

    assertThrows(NullPointerException.class, supplier::get);
  }
}
