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

import com.google.common.testing.GcFinalization;
import java.lang.ref.WeakReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link MemoizingInterruptibleSupplier}. */
@RunWith(JUnit4.class)
public final class MemoizingInterruptibleSupplierTest {

  private int callCount = 0;
  private String returnVal = "";
  private CallCounter callCounter = new CallCounter();

  private final class CallCounter {
    public String call() {
      ++callCount;
      return returnVal;
    }
  }

  @Test
  public void getReturnsCorrectResult() throws Exception {
    MemoizingInterruptibleSupplier<String> supplier =
        MemoizingInterruptibleSupplier.of(callCounter::call);
    returnVal = "abc";

    String result = supplier.get();

    assertThat(result).isEqualTo("abc");
  }

  @Test
  public void subsequentCallToGetReturnsCorrectResult() throws Exception {
    MemoizingInterruptibleSupplier<String> supplier =
        MemoizingInterruptibleSupplier.of(callCounter::call);
    returnVal = "abc";

    supplier.get();
    String result = supplier.get();

    assertThat(result).isEqualTo("abc");
  }

  @Test
  public void onlyCallsDelegateOnce() throws Exception {
    MemoizingInterruptibleSupplier<String> supplier =
        MemoizingInterruptibleSupplier.of(callCounter::call);

    supplier.get();
    supplier.get();

    assertThat(callCount).isEqualTo(1);
  }

  @Test
  public void freesReferenceToDelegeteAfterGet() throws Exception {
    MemoizingInterruptibleSupplier<String> supplier =
        MemoizingInterruptibleSupplier.of(callCounter::call);
    WeakReference<Object> ref = new WeakReference<>(callCounter);
    callCounter = null;

    supplier.get();

    GcFinalization.awaitClear(ref);
  }

  @Test
  public void notInitializedBeforeCallingGet() {
    MemoizingInterruptibleSupplier<String> supplier =
        MemoizingInterruptibleSupplier.of(callCounter::call);

    boolean initialized = supplier.isInitialized();

    assertThat(initialized).isFalse();
  }

  @Test
  public void isInitializedAfterCallingGet() throws Exception {
    MemoizingInterruptibleSupplier<String> supplier =
        MemoizingInterruptibleSupplier.of(callCounter::call);

    supplier.get();
    boolean initialized = supplier.isInitialized();

    assertThat(initialized).isTrue();
  }

  @Test
  public void isStillInitializedAfterSubsequentCallToGet() throws Exception {
    MemoizingInterruptibleSupplier<String> supplier =
        MemoizingInterruptibleSupplier.of(callCounter::call);

    supplier.get();
    supplier.get();
    boolean initialized = supplier.isInitialized();

    assertThat(initialized).isTrue();
  }

  @Test
  public void of_returnsSameInstanceIfAlreadyMemoizing() {
    InterruptibleSupplier<String> supplier = MemoizingInterruptibleSupplier.of(callCounter::call);
    assertThat(MemoizingInterruptibleSupplier.of(supplier)).isSameInstanceAs(supplier);
  }
}
