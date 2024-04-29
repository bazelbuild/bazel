// Copyright 2024 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.common.testing.EqualsTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ValueOrExceptionTest {

  @Test
  public void factoryMethods_requireNonNull() {
    assertThrows(NullPointerException.class, () -> ValueOrException.ofValue(null));
    assertThrows(NullPointerException.class, () -> ValueOrException.ofException(null));
  }

  @Test
  public void isPresent_basicBehavior() {
    assertThat(ValueOrException.ofValue(new TestValue(123)).isPresent()).isTrue();
    assertThat(
            ValueOrException.ofValue(new TestException("error") /* as value, not as exception */)
                .isPresent())
        .isTrue();
    assertThat(ValueOrException.ofException(new TestException("error")).isPresent()).isFalse();
  }

  @Test
  public void get_ofValue_succeeds() throws Exception {
    TestValue value = new TestValue(42);
    ValueOrException<TestValue, TestException> valueOrException = ValueOrException.ofValue(value);
    assertThat(valueOrException.get()).isSameInstanceAs(value);
    assertThat(valueOrException.getUnchecked()).isSameInstanceAs(value);
  }

  @Test
  public void get_ofException_throws() {
    TestException exception = new TestException("i/o error");
    ValueOrException<TestValue, TestException> valueOrException =
        ValueOrException.ofException(exception);
    assertThat(assertThrows(TestException.class, valueOrException::get))
        .isSameInstanceAs(exception);
    assertThat(assertThrows(IllegalStateException.class, valueOrException::getUnchecked))
        .hasCauseThat()
        .isSameInstanceAs(exception);
  }

  @Test
  public void getException_basicBehavior() {
    TestValue value = new TestValue(42);
    TestException exception = new TestException("i/o error");
    assertThrows(IllegalStateException.class, () -> ValueOrException.ofValue(value).getException());
    assertThrows(
        IllegalStateException.class,
        () -> ValueOrException.ofValue(exception /* as value, not as exception */).getException());
    assertThat(ValueOrException.ofException(exception).getException()).isSameInstanceAs(exception);
  }

  @Test
  public void toString_basicFunctionality() {
    assertThat(ValueOrException.ofValue(new TestValue(42)).toString())
        .isEqualTo("ValueOrException.OfValue[TestValue(42)]");
    assertThat(ValueOrException.ofValue(new TestException("failure")).toString())
        .isEqualTo("ValueOrException.OfValue[TestException('failure')]");
    assertThat(ValueOrException.ofException(new TestException("failure")).toString())
        .isEqualTo("ValueOrException.OfException[TestException('failure')]");
  }

  @Test
  public void hashCode_basicFunctionality() {
    int unused = ValueOrException.ofValue(new TestValue(42)).hashCode(); // Should not throw.
    int unused2 =
        ValueOrException.ofException(new TestException("fail")).hashCode(); // Should not throw.
  }

  @Test
  public void equals() {
    TestValue value12345 = new TestValue(12345);
    TestException failure = new TestException("failure");

    new EqualsTester()
        .addEqualityGroup(
            ValueOrException.ofValue(value12345), ValueOrException.ofValue(new TestValue(12345)))
        .addEqualityGroup(ValueOrException.ofValue(new TestValue(12346)))
        .addEqualityGroup(
            ValueOrException.ofException(failure),
            ValueOrException.ofException(new TestException("failure")))
        .addEqualityGroup(ValueOrException.ofValue(failure /* as _value_, not exception! */))
        .addEqualityGroup(ValueOrException.ofException(new TestException("other failure")))
        .testEquals();
  }

  private static final class TestValue {
    private final int content;

    TestValue(int content) {
      this.content = content;
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof TestValue testValue) {
        return testValue.content == content;
      } else {
        return false;
      }
    }

    @Override
    public int hashCode() {
      return Integer.valueOf(content).hashCode();
    }

    @Override
    public String toString() {
      return String.format("TestValue(%d)", content);
    }
  }

  @SuppressWarnings("OverrideThrowableToString") // toString() overridden for testing
  private static final class TestException extends Exception {
    TestException(String message) {
      super(message);
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof TestException testException) {
        return testException.getMessage().equals(getMessage());
      } else {
        return false;
      }
    }

    @Override
    public int hashCode() {
      return getMessage().hashCode();
    }

    @Override
    public String toString() {
      return String.format("TestException('%s')", getMessage());
    }
  }
}
