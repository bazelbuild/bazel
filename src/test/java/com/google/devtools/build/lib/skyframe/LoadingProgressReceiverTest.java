// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.base.Supplier;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/**
 * Tests {@link LoadingProgressReceiver}
 */
@RunWith(JUnit4.class)
public class LoadingProgressReceiverTest {

  @Test
  public void testPackageVisible() {
    // If there is only one package being loaded, it should be visible
    // somewhere in the progress string.
    LoadingProgressReceiver progressReceiver = new LoadingProgressReceiver();

    String packageName = "//some/fancy/package/name";
    progressReceiver.enqueueing(SkyKey.create(SkyFunctions.PACKAGE, packageName));
    String state = progressReceiver.progressState();

    assertTrue(
        "Package name '" + packageName + "' should occour in state '" + state + "'",
        state.contains(packageName));
  }

  @Test
  public void testFinishedPackage() {
    // Consider the situation where two packages are loaded, and the one that
    // started loading second has already finished loading. ...
    LoadingProgressReceiver progressReceiver = new LoadingProgressReceiver();

    String packageName = "//some/fancy/package/name";
    String otherPackageName = "//some/other/unrelated/package";
    SkyKey otherKey = SkyKey.create(SkyFunctions.PACKAGE, otherPackageName);
    Supplier<SkyValue> valueSupplier = Mockito.mock(Supplier.class);
    progressReceiver.enqueueing(SkyKey.create(SkyFunctions.PACKAGE, packageName));
    progressReceiver.enqueueing(otherKey);
    progressReceiver.evaluated(
        otherKey, valueSupplier, EvaluationProgressReceiver.EvaluationState.BUILT);
    String state = progressReceiver.progressState();

    // ... Then, the finished package is not mentioned in the state,
    assertFalse(
        "Package name '" + otherPackageName + "' should not occur in state '" + state + "'",
        state.contains(otherPackageName));
    // the only running package is mentioned in the state, and
    assertTrue(
        "Package name '" + packageName + "' should occour in state '" + state + "'",
        state.contains(packageName));
    // the progress count is correct.
    assertTrue("Progress count should occur in state", state.contains("1 / 2"));
  }
}
