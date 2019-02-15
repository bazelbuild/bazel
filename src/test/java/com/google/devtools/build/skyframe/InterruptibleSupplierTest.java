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
package com.google.devtools.build.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.testing.GcFinalization;
import java.lang.ref.WeakReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link InterruptibleSupplier}. */
@RunWith(JUnit4.class)
public final class InterruptibleSupplierTest {

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
  public void memoize_returnsCorrectResult() throws Exception {
    InterruptibleSupplier<String> supplier = InterruptibleSupplier.Memoize.of(callCounter::call);
    returnVal = "abc";

    String result = supplier.get();

    assertThat(result).isEqualTo("abc");
  }

  @Test
  public void memoize_onlyCallsDelegateOnce() throws Exception {
    InterruptibleSupplier<String> supplier = InterruptibleSupplier.Memoize.of(callCounter::call);

    supplier.get();
    supplier.get();

    assertThat(callCount).isEqualTo(1);
  }

  @Test
  public void memoize_freesReferenceToDelegeteAfterGet() throws Exception {
    InterruptibleSupplier<String> supplier = InterruptibleSupplier.Memoize.of(callCounter::call);
    WeakReference<Object> ref = new WeakReference<>(callCounter);
    callCounter = null;

    supplier.get();

    GcFinalization.awaitClear(ref);
  }
}
