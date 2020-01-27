// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.common.options;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link GenericTypeHelper}.
 */
@RunWith(JUnit4.class)
public class GenericTypeHelperTest {

  private static interface DoSomething<T> {
    T doIt();
  }

  private static class StringSomething implements DoSomething<String> {
    @Override
    public String doIt() {
      return null;
    }
  }

  private static class EnumSomething<T> implements DoSomething<T> {
    @Override
    public T doIt() {
      return null;
    }
  }

  private static class AlphabetSomething extends EnumSomething<String> {
  }

  private static class AlphabetTwoSomething extends AlphabetSomething {
  }

  private static void assertDoIt(Class<?> expected,
      Class<? extends DoSomething<?>> implementingClass) throws Exception {
    assertThat(
            GenericTypeHelper.getActualReturnType(
                implementingClass, implementingClass.getMethod("doIt")))
        .isEqualTo(expected);
  }

  @Test
  public void getConverterType() throws Exception {
    assertDoIt(String.class, StringSomething.class);
  }

  @Test
  public void getConverterTypeForGenericExtension() throws Exception {
    assertDoIt(String.class, AlphabetSomething.class);
  }

  @Test
  public void getConverterTypeForGenericExtensionSecondGrade() throws Exception {
    assertDoIt(String.class, AlphabetTwoSomething.class);
  }
}
