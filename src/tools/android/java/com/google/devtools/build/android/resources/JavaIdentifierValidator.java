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
package com.google.devtools.build.android.resources;

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableSet;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/** Validates resource identifiers and packages for java identifier validity. */
public class JavaIdentifierValidator {

  private JavaIdentifierValidator() {}

  /** Thrown when a resource filed is not a valida java identifier. */
  public static class InvalidJavaIdentifier extends RuntimeException {

    /** Creates a new exception. */
    public InvalidJavaIdentifier(String message) {
      super(message);
    }
  }

  /**
   * Validates a resource identifier for java correctness.
   *
   * @param identifier an identifier derived from an android resource.
   * @param additionalInformation optional information about the identifier.
   * @return The identifier if valid.
   * @throws InvalidJavaIdentifier if the identifier is invalid.
   */
  public static String validate(String identifier, Object... additionalInformation) {
    if (VALID_JAVA_IDENTIFIER.test(identifier)) {
      return identifier;
    }
    throw new InvalidJavaIdentifier(
        String.format(
            "%s is an invalid java identifier %s.",
            identifier,
            Stream.of(additionalInformation).map(Object::toString).collect(joining(" "))));
  }

  private static final ImmutableSet<String> JAVA_RESERVED =
      ImmutableSet.of(
          "abstract",
          "assert",
          "boolean",
          "break",
          "byte",
          "case",
          "catch",
          "char",
          "class",
          "const",
          "continue",
          "default",
          "double",
          "do",
          "else",
          "enum",
          "extends",
          "false",
          "final",
          "finally",
          "float",
          "for",
          "goto",
          "if",
          "implements",
          "import",
          "instanceof",
          "int",
          "interface",
          "long",
          "native",
          "new",
          "null",
          "package",
          "private",
          "protected",
          "public",
          "return",
          "short",
          "static",
          "strictfp",
          "super",
          "switch",
          "synchronized",
          "this",
          "throw",
          "throws",
          "transient",
          "true",
          "try",
          "void",
          "volatile",
          "while");

  private static final Predicate<String> VALID_JAVA_IDENTIFIER =
      ((Predicate<String>) JAVA_RESERVED::contains)
          .negate()
          .and(Pattern.compile("^([a-zA-Z_$][a-zA-Z\\d_$]*)$").asPredicate());
}
