// Copyright 2015 The Bazel Authors. All rights reserved.
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


import java.io.IOException;
import java.lang.annotation.*;

class A {

  // Exercise removal of class initializers:
  static { }

  // Exercise removal of private members:
  private String privateField;

  protected int protectedField;

  // Exercise ConstantValue annotation
  public static final long L1 = 123L;

  // Exercise removal of private members:
  private void privateMethod() {
    System.err.println("foofoofoofoo");
  }

  // Exercise Signature annotation:
  protected <T> T protectedMethod(T t) { return t; }

  // Exercise Deprecated annotation:
  @Deprecated
  // Exercise Exceptions annotation:
  public void deprecatedMethod() throws IOException {}

  // Exercise retention of private inner classes:
  private class PrivateInner {}

  // Exercise InnerClasses attribute:
  public class PublicInner {}

  public @interface MyAnnotation {
    // Exercise BaseTypeElementValue:
    String a() default "foo";

    // Exercise EnumTypeElementValue:
    ElementType b() default ElementType.METHOD;

    // Exercise ClassTypeElementValue:
    Class<?> c() default String.class;

    // Exercise ArrayTypeElementValue:
    String[] d() default { "foo", "bar" };

    // Exercise AnnotationTypeElementValue:
    Deprecated e() default @Deprecated;
  }

  @Retention(RetentionPolicy.RUNTIME)
  public @interface RuntimeAnnotation {}

}
