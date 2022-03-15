// Copyright 2022 The Bazel Authors. All rights reserved.
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
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bugreport.BugReporter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;

/** Tests for {@link GraphTraversingHelper}. */
@RunWith(JUnit4.class)
public final class GraphTraversingHelperTest {
  SkyFunction.Environment mockEnv = mock(SkyFunction.Environment.class);
  SkyKey keyA = mock(SkyKey.class);
  SkyKey keyB = mock(SkyKey.class);
  SomeErrorException exn = new SomeErrorException("");
  SkyValue value = mock(SkyValue.class);

  private static final class SomeOtherErrorException extends Exception {
    private SomeOtherErrorException() {}
  }

  @Test
  public void declareDependenciesAndCheckIfValuesMissing_valuesMissingBeforeCompute()
      throws Exception {
    when(mockEnv.valuesMissing()).thenReturn(true);
    when(mockEnv.getOrderedValuesAndExceptions(ImmutableSet.of(keyA))).thenReturn(null);
    boolean valuesMissing =
        GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissing(
            mockEnv, ImmutableSet.of(keyA), SomeOtherErrorException.class);
    assertThat(valuesMissing).isTrue();
  }

  @Test
  public void declareDependenciesAndCheckIfValuesMissing_valuesMissingAfterCompute()
      throws Exception {
    BugReporter mockReporter = mock(BugReporter.class);
    SkyframeIterableResult result =
        new SkyframeIterableResult(
            () -> {}, ImmutableSet.of(ValueOrUntypedException.ofExn(exn)).iterator());
    when(mockEnv.getOrderedValuesAndExceptions(ImmutableSet.of(keyA))).thenReturn(result);
    boolean valuesMissing =
        GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissing(
            mockEnv,
            ImmutableSet.of(keyA),
            SomeOtherErrorException.class,
            /*exceptionClass2=*/ null,
            mockReporter);
    ArgumentCaptor<IllegalStateException> exceptionCaptor =
        ArgumentCaptor.forClass(IllegalStateException.class);
    verify(mockReporter).sendBugReport(exceptionCaptor.capture());
    assertThat(exceptionCaptor.getValue()).hasMessageThat().contains("Some value from");
    verifyNoMoreInteractions(mockReporter);
    assertThat(valuesMissing).isTrue();
  }

  @Test
  public void declareDependenciesAndCheckIfValuesMissing_notValuesMissingAfterCompute()
      throws Exception {
    SkyframeIterableResult result =
        new SkyframeIterableResult(
            () -> {},
            ImmutableList.of(
                    ValueOrUntypedException.ofExn(exn),
                    ValueOrUntypedException.ofValueUntyped(value))
                .iterator());
    when(mockEnv.getOrderedValuesAndExceptions(ImmutableList.of(keyA, keyB))).thenReturn(result);
    boolean valuesMissing =
        GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissing(
            mockEnv, ImmutableList.of(keyA, keyB), SomeErrorException.class);
    assertThat(valuesMissing).isFalse();
  }
}
