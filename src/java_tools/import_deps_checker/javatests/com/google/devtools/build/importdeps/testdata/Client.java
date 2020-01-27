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
import com.google.devtools.build.importdeps.testdata.LibraryAnnotations.AnnotationFlag;
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
@ClassAnnotation(
  friends = {Library.class, LibraryException.class},
  nested = @SuppressWarnings({"unused", "unchecked"})
)
public class Client<@TypeAnnotation T> extends Library
    implements LibraryInterface, LibraryInterface.One, LibraryInterface.Two {

  @SuppressWarnings("unused")
  @FieldAnnotation
  private Library.Class1 field;

  @SuppressWarnings("unused")
  @FieldAnnotation({1, 2, 3})
  private LibraryAnnotations annotations;

  public static final Class1 I = Class1.I;

  @ConstructorAnnotation
  public Client() {}

  @MethodAnnotation(name = "method")
  public void method(@ParameterAnnotation(position = 0) int p, Library.Class2 p2)
      throws LibraryException {
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
    Object[] copy = array.clone();
    array = (Class8[][]) copy;
    System.out.println(array.clone().length);
    Integer b = Integer.valueOf(0);
    System.out.println(b);

    Class11 eleven = new Class11();
    eleven.foo();
    eleven.bar();

    // Regression test for b/123020654: check class literals
    Class<?> class12 = Class12.class;
  }

  public void testEnums() {
    EnumTest a = EnumTest.A;
    System.out.println(a.ordinal());
    System.out.println(a.name());
  }

  /** An inner annotation. */
  @Retention(RetentionPolicy.RUNTIME)
  @Target(ElementType.TYPE)
  @AnnotationAnnotation(AnnotationFlag.Y)
  public @interface NestedAnnotation {}

  @Override
  public void callOne() {
    // Do nothing.
  }

  @Override
  public void callTwo() {
    // Do nothing.
  }

  public enum EnumTest {
    A,
    B,
    C
  }

  private class InnerClassWithSyntheticConstructorParam {
    // This constructor has a synthetic parameter for the outer object (b/78024300).  If there are
    // parameter annotations then ASM generates "java/lang/Synthetic" annotations on the synthetic
    // parameters, but java/lang/Synthetic doesn't exist!
    InnerClassWithSyntheticConstructorParam(@ParameterAnnotation(position = 2) int p) {}

    @Override
    public String toString() {
      return String.valueOf(field);
    }
  }
}
