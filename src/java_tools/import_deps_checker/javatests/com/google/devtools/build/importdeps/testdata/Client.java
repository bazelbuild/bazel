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
package com.google.devtools.build.importdeps.testdata;

import com.google.devtools.build.importdeps.testdata.LibraryAnnotations.AnnotationAnnotation;
import com.google.devtools.build.importdeps.testdata.LibraryAnnotations.ClassAnnotation;
import com.google.devtools.build.importdeps.testdata.LibraryAnnotations.ConstructorAnnotation;
import com.google.devtools.build.importdeps.testdata.LibraryAnnotations.FieldAnnotation;
import com.google.devtools.build.importdeps.testdata.LibraryAnnotations.MethodAnnotation;
import com.google.devtools.build.importdeps.testdata.LibraryAnnotations.ParameterAnnotation;
import com.google.devtools.build.importdeps.testdata.LibraryAnnotations.TypeAnnotation;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.util.Objects;

/** Client class that uses several libraries. */
@ClassAnnotation
public class Client<@TypeAnnotation T> extends Library implements LibraryInterface {

  @SuppressWarnings("unused")
  @FieldAnnotation
  private Library.Class1 field;

  @SuppressWarnings("unused")
  @FieldAnnotation
  private LibraryAnnotations annotations;

  public static final Class1 I = Class1.I;

  @ConstructorAnnotation
  public Client() {}

  @MethodAnnotation
  public void method(@ParameterAnnotation int p, Library.Class2 p2) throws LibraryException {
    Objects.nonNull(p2); // javac9 silently uses Objects.
    Class3 c3 = new Class3();
    Class4 c4 = c3.field;
    c3.field = c4;
    Func<Class5> func = c4::createClass5;
    Class5 class5 = func.get();
    @SuppressWarnings("unused")
    Class6 class6 = class5.create(new Class7());
    @SuppressWarnings("unused")
    Class8[][] array = new Class8[10][10];
    Class9[] array2 = new Class9[10];
    array2[0] = new Class10();
  }

  /** An inner annotation. */
  @Retention(RetentionPolicy.RUNTIME)
  @Target(ElementType.TYPE)
  @AnnotationAnnotation
  public @interface NestedAnnotation {}
}
