// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar.testdata.java8;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/** Test for b/38302860. The annotations of default methods should be kept after desugaring. */
public class AnnotationsOfDefaultMethodsShouldBeKept {

  /**
   * An interface, that has annotation, annotated abstract methods, and annotated default methods.
   * After desugaring, all these annotations should remain in the interface.
   */
  @SomeAnnotation(1)
  public interface AnnotatedInterface {

    @SomeAnnotation(2)
    void annotatedAbstractMethod();

    @SomeAnnotation(3)
    default void annotatedDefaultMethod() {}
  }

  /** A simple annotation, used for testing. */
  @Retention(value = RetentionPolicy.RUNTIME)
  public @interface SomeAnnotation {
    int value();
  }
}
