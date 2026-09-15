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
 * A class with various deprecated parts used for testing.
 */
@SuppressWarnings("dep-ann")
final class DeprecatedParts {
  @Deprecated
  public static String deprecatedViaAnnotationStaticField = "hi";

  /**
   * @deprecated Marked deprecated for test.
   */
  public static String deprecatedViaJavadocStaticField = "hi";

  @Deprecated
  public String deprecatedViaAnnotationField = "hi";

  /**
   * @deprecated Marked deprecated for test.
   */
  public String deprecatedViaJavadocField = "hi";

  @Deprecated
  public void deprecatedViaAnnotationMethod() {
  }

  /**
   * @deprecated Marked deprecated for test.
   */
  public void deprecatedViaJavadocMethod() {
  }

  // Note: @Deprecated can be applied to parameters, but the compiler doesn't seem to use it

  @Deprecated
  public static void deprecatedViaAnnotationStaticMethod() {
  }

  /**
   * @deprecated Marked deprecated for test.
   */
  public static void deprecatedViaJavadocStaticMethod() {
  }

  @Deprecated
  public @interface DeprecatedViaAnnotationAnnotation {}

  /**
   * @deprecated Marked deprecated for test.
   */
  public @interface DeprecatedViaJavadocAnnotation {}

  @Deprecated
  public static class DeprecatedViaAnnotationClass {}

  /**
   * @deprecated Marked deprecated to exercise preservation of deprecation.
   */
  public static class DeprecatedViaJavadocClass {}

  public static class DeprecatedViaAnnotationConstructor {
    @Deprecated
    public DeprecatedViaAnnotationConstructor() {}
  }

  public static class DeprecatedViaJavadocConstructor {
    /**
     * @deprecated Marked deprecated for test.
     */
    public DeprecatedViaJavadocConstructor() {}
  }

  @Deprecated
  public interface DeprecatedViaAnnotationInterface {}

  /**
   * @deprecated Marked deprecated for test.
   */
  public interface DeprecatedViaJavadocInterface {}
}
