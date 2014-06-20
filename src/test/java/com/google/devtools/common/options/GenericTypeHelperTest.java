// Copyright 2014 Google Inc. All rights reserved.
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

import junit.framework.TestCase;

/**
 * Tests {@link GenericTypeHelper}.
 */
public class GenericTypeHelperTest extends TestCase {

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
    assertEquals(expected,
        GenericTypeHelper.getActualReturnType(implementingClass,
            implementingClass.getMethod("doIt")));
  }

  public void testGetConverterType() throws Exception {
    assertDoIt(String.class, StringSomething.class);
  }

  public void testGetConverterTypeForGenericExtension() throws Exception {
    assertDoIt(String.class, AlphabetSomething.class);
  }

  public void testGetConverterTypeForGenericExtensionSecondGrade() throws Exception {
    assertDoIt(String.class, AlphabetTwoSomething.class);
  }
}
