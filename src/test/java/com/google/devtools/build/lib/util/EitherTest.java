// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.testing.EqualsTester;
import java.util.function.Consumer;
import java.util.function.Function;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link Either}.
 */
@RunWith(JUnit4.class)
public class EitherTest {
  @SuppressWarnings("unchecked")
  @Test
  public void leftConsume() {
    Either<Integer, String> underTest = Either.ofLeft(42);

    Consumer<Integer> mockIntegerConsumer = mock(Consumer.class);
    Consumer<String> mockStringConsumer = mock(Consumer.class);
    underTest.consume(mockIntegerConsumer, mockStringConsumer);

    verify(mockIntegerConsumer, times(1)).accept(eq(42));
    verify(mockStringConsumer, never()).accept(any());
  }

  @SuppressWarnings("unchecked")
  @Test
  public void leftMap() {
    Either<Integer, String> underTest = Either.ofLeft(42);

    Function<Integer, Boolean> mockIntegerFunction = mock(Function.class);
    Function<String, Boolean> mockStringFunction = mock(Function.class);
    when(mockIntegerFunction.apply(eq(42))).thenReturn(true);
    assertThat(underTest.map(mockIntegerFunction, mockStringFunction)).isTrue();

    verify(mockIntegerFunction, times(1)).apply(eq(42));
    verify(mockStringFunction, never()).apply(any());
  }

  @SuppressWarnings("unchecked")
  @Test
  public void rightConsume() {
    Either<Integer, String> underTest = Either.ofRight("cat");

    Consumer<Integer> mockIntegerConsumer = mock(Consumer.class);
    Consumer<String> mockStringConsumer = mock(Consumer.class);
    underTest.consume(mockIntegerConsumer, mockStringConsumer);

    verify(mockIntegerConsumer, never()).accept(any());
    verify(mockStringConsumer, times(1)).accept(eq("cat"));
  }

  @SuppressWarnings("unchecked")
  @Test
  public void rightMap() {
    Either<Integer, String> underTest = Either.ofRight("cat");

    Function<Integer, Boolean> mockIntegerFunction = mock(Function.class);
    Function<String, Boolean> mockStringFunction = mock(Function.class);
    when(mockStringFunction.apply(eq("cat"))).thenReturn(true);
    assertThat(underTest.map(mockIntegerFunction, mockStringFunction)).isTrue();

    verify(mockIntegerFunction, never()).apply(any());
    verify(mockStringFunction, times(1)).apply(eq("cat"));
  }

  @Test
  public void equalsAndHashCode() {
    new EqualsTester()
        .addEqualityGroup(Either.ofLeft(null), Either.ofLeft(null))
        .addEqualityGroup(Either.ofLeft(1), Either.ofLeft(1))
        .addEqualityGroup(Either.ofLeft(2), Either.ofLeft(2))
        .addEqualityGroup(Either.ofRight(1), Either.ofRight(1))
        .addEqualityGroup(Either.ofRight("cat"), Either.ofRight("cat"))
        .addEqualityGroup(Either.ofRight("dog"), Either.ofRight("dog"))
        .testEquals();
  }
}
