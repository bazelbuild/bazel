/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.typeannotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/** The test source class for testing annotation desugaring. */
public class AnnotationUser {

  @SuppressWarnings("unused") // Test source
  public void localVarWithTypeUseAnnotation(List<String> inTextList) {
    List<@EnhancedType String> localTextList = inTextList;
  }

  public void instructionTypeUseAnnotation() {
    new ArrayList<@EnhancedType String>();
  }

  public void tryCatchTypeAnnotation(RuntimeException inException) {
    try {
      throw inException;
    } catch (@EnhancedType IllegalArgumentException e) {
      throw new UnsupportedOperationException(e);
    }
  }

  // @EnhancedVar is expected to be absent in Javac-compiled bytecode, even through @EnhancedVar is
  // declared with a runtime retention policy. This test case ensures any retained annotation use
  // within a method body is a type annotation.
  @SuppressWarnings("unused") // Test source
  public Function<Integer, Integer> localNonTypeAnnotations(String inputText) {
    @EnhancedVar String localText = inputText;
    try {
      return (@EnhancedVar Integer x) -> 2 * x;
    } catch (@EnhancedVar IllegalStateException e) {
      throw new UnsupportedOperationException(e);
    }
  }

  @Target({ElementType.TYPE_USE, ElementType.TYPE_PARAMETER})
  @Retention(RetentionPolicy.RUNTIME)
  @interface EnhancedType {}

  @Retention(RetentionPolicy.RUNTIME)
  @interface EnhancedVar {}
}
