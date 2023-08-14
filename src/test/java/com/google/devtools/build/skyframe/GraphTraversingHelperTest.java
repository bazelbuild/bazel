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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bugreport.BugReporter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link GraphTraversingHelper}. */
@RunWith(JUnit4.class)
public final class GraphTraversingHelperTest {

  private final SkyFunction.Environment mockEnv = mock(SkyFunction.Environment.class);
  private final SkyKey keyA = mock(SkyKey.class, "keyA");
  private final SkyKey keyB = mock(SkyKey.class, "keyB");
  private final SomeErrorException exn = new SomeErrorException("");
  private final SkyValue value = mock(SkyValue.class);

  private static final class SomeOtherErrorException extends Exception {
    private SomeOtherErrorException() {}
  }

  @Test
  public void declareDependenciesAndCheckIfValuesMissing_valuesMissingBeforeCompute()
      throws Exception {
    when(mockEnv.valuesMissing()).thenReturn(true);
    when(mockEnv.getValuesAndExceptions(ImmutableSet.of(keyA))).thenReturn(null);
    boolean valuesMissing =
        GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissing(
            mockEnv, ImmutableSet.of(keyA), SomeOtherErrorException.class);
    assertThat(valuesMissing).isTrue();
  }

  @Test
  public void declareDependenciesAndCheckIfValuesMissing_valuesMissingAfterCompute()
      throws Exception {
    BugReporter mockReporter = mock(BugReporter.class);
    SkyframeLookupResult result =
        new SimpleSkyframeLookupResult(
            () -> {}, ImmutableMap.of(keyA, ValueOrUntypedException.ofExn(exn))::get);
    when(mockEnv.getValuesAndExceptions(ImmutableSet.of(keyA))).thenReturn(result);
    boolean valuesMissing =
        GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissing(
            mockEnv,
            ImmutableSet.of(keyA),
            SomeOtherErrorException.class,
            /*exceptionClass2=*/ null,
            mockReporter);
    verify(mockReporter)
        .logUnexpected("Value for: '%s' was missing, this should never happen", keyA);
    verifyNoMoreInteractions(mockReporter);
    assertThat(valuesMissing).isTrue();
  }

  @Test
  public void declareDependenciesAndCheckIfValuesMissing_notValuesMissingAfterCompute()
      throws Exception {
    SkyframeLookupResult result =
        new SimpleSkyframeLookupResult(
            () -> {},
            ImmutableMap.of(
                    keyA,
                    ValueOrUntypedException.ofExn(exn),
                    keyB,
                    ValueOrUntypedException.ofValueUntyped(value))
                ::get);
    when(mockEnv.getValuesAndExceptions(ImmutableList.of(keyA, keyB))).thenReturn(result);
    boolean valuesMissing =
        GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissing(
            mockEnv, ImmutableList.of(keyA, keyB), SomeErrorException.class);
    assertThat(valuesMissing).isFalse();
  }

  @Test
  public void declareDependenciesAndCheckIfValuesMissing_nullAfterError_hasCorrectKeyInBugReport()
      throws Exception {
    SkyframeLookupResult result =
        new SimpleSkyframeLookupResult(
            () -> {},
            ImmutableMap.of(
                    keyA,
                    ValueOrUntypedException.ofExn(exn),
                    keyB,
                    ValueOrUntypedException.ofNull())
                ::get);
    when(mockEnv.getValuesAndExceptions(ImmutableList.of(keyA, keyB))).thenReturn(result);
    BugReporter mockReporter = mock(BugReporter.class);

    boolean valuesMissing =
        GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissing(
            mockEnv, ImmutableList.of(keyA, keyB), SomeErrorException.class, null, mockReporter);

    assertThat(valuesMissing).isTrue();
    verify(mockReporter)
        .logUnexpected("Value for: '%s' was missing, this should never happen", keyB);
    verifyNoMoreInteractions(mockReporter);
  }

  @Test
  public void declareDependenciesAndCheckIfValuesMissingMaybeWithExceptions_beforeCompute()
      throws Exception {
    when(mockEnv.valuesMissing()).thenReturn(true);
    when(mockEnv.getValuesAndExceptions(ImmutableSet.of(keyB))).thenReturn(null);

    assertThat(
            GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissingMaybeWithExceptions(
                mockEnv, ImmutableSet.of(keyB)))
        .isTrue();
  }

  @Test
  public void declareDependenciesAndCheckIfValuesMissingMaybeWithExceptions_valuesMissing()
      throws Exception {
    when(mockEnv.getValuesAndExceptions(ImmutableSet.of(keyA)))
        .thenReturn(
            new SimpleSkyframeLookupResult(
                () -> {}, ImmutableMap.of(keyA, ValueOrUntypedException.ofExn(exn))::get));

    assertThat(
            GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissingMaybeWithExceptions(
                mockEnv, ImmutableSet.of(keyA)))
        .isTrue();
  }

  @Test
  public void declareDependenciesAndCheckIfValuesMissingMaybeWithExceptions_notValuesMissing()
      throws Exception {
    when(mockEnv.getValuesAndExceptions(ImmutableSet.of(keyB)))
        .thenReturn(
            new SimpleSkyframeLookupResult(
                () -> {},
                ImmutableMap.of(keyB, ValueOrUntypedException.ofValueUntyped(value))::get));

    assertThat(
            GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissingMaybeWithExceptions(
                mockEnv, ImmutableSet.of(keyB)))
        .isFalse();
  }
}
