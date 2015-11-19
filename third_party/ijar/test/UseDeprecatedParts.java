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

/**
 * A class that uses deprecated things for testing.
 */
final class UseDeprecatedParts {
  @DeprecatedParts.DeprecatedViaAnnotationAnnotation String annotatedField1;
  @DeprecatedParts.DeprecatedViaJavadocAnnotation String annotatedField2;

  static final DeprecatedParts.DeprecatedViaAnnotationConstructor dummyField1 =
      new DeprecatedParts.DeprecatedViaAnnotationConstructor();
  static final DeprecatedParts.DeprecatedViaJavadocConstructor dummyField2 =
      new DeprecatedParts.DeprecatedViaJavadocConstructor ();

  public static void dummyMethod() {
    System.out.println(DeprecatedParts.deprecatedViaAnnotationStaticField);
    System.out.println(DeprecatedParts.deprecatedViaJavadocStaticField);

    System.out.println(new DeprecatedParts().deprecatedViaAnnotationField);
    System.out.println(new DeprecatedParts().deprecatedViaJavadocField);

    new DeprecatedParts().deprecatedViaAnnotationMethod();
    new DeprecatedParts().deprecatedViaJavadocMethod();

    DeprecatedParts.deprecatedViaAnnotationStaticMethod();
    DeprecatedParts.deprecatedViaJavadocStaticMethod();
  }

  class Class1 extends DeprecatedParts.DeprecatedViaAnnotationClass {}

  class Class2 extends DeprecatedParts.DeprecatedViaJavadocClass {}

  interface Interface1 extends DeprecatedParts.DeprecatedViaAnnotationInterface {}

  interface Interface2 extends DeprecatedParts.DeprecatedViaJavadocInterface {}
}
