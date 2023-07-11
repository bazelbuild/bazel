// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Verify.verify;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import java.util.Arrays;

/**
 * Parses the class signature as specified by <a
 * href="https://docs.oracle.com/javase/specs/jvms/se12/html/jvms-4.html#jvms-ClassSignature">$4.7.9.1</a>
 * of the Java Virtual Machine Specification.
 */
public final class ClassSignatureParser {

  private final String signature;
  private final String[] interfaceTypeParameters;
  private int index;
  private int numInterfaces;
  private String generics;
  private SuperclassSignature superclassSignature;

  /**
   * Rudimentary parser for class signatures as specified by <a
   * href="https://docs.oracle.com/javase/specs/jvms/se12/html/jvms-4.html#jvms-ClassSignature">$4.7.9.1</a>.
   * It returns an object containing the generic parameters, super class and any other generic
   * interface parameters. The number of generic interface parameters is equal to the {@code
   * expectedNumInterfaces}.
   *
   * <p>For example, the signature {@code
   * <K:Ljava/lang/Object;V:Ljava/lang/Object;>Ljava/lang/Object;Ljava/util/Map<TK;TV;>;} will
   * result in:
   *
   * <pre>{@code
   * {
   *   typeParameters: "<K:Ljava/lang/Object;V:Ljava/lang/Object;>",
   *   superClassSignature: {
   *     identifier: "Ljava/lang/Object;",
   *     typeParameters: ""
   *   },
   *   interfaceTypeParameters: ["TK;TV;"]
   * }
   * }</pre>
   *
   * @param name The identifier of the class
   * @param signature The class signature as specified in <a
   *     href="https://docs.oracle.com/javase/specs/jvms/se12/html/jvms-4.html#jvms-ClassSignature">$4.7.9.1</a>
   * @param superName The identifier of the superclass that it extends associated by the {@code
   *     signature}
   * @param interfaces The interfaces that are being implemented by the class associated by the
   *     {@code signature}
   * @return An object containing the generic parameters, super class and any other generic
   *     interface parameters
   */
  static ClassSignature readTypeParametersForInterfaces(
      String name, String signature, String superName, String[] interfaces) {

    int expectedNumInterfaces = interfaces.length;

    try {
      ClassSignatureParser classSignatureParser =
          new ClassSignatureParser(signature, expectedNumInterfaces);
      classSignatureParser.readFromSignature();

      verify(
          classSignatureParser.numInterfaces == expectedNumInterfaces,
          "Incorrect number of generic parameters parsed. Got %s interfaces, but expected %s",
          classSignatureParser.numInterfaces,
          expectedNumInterfaces);

      return ClassSignature.create(classSignatureParser);

    } catch (RuntimeException e) {
      throw new IllegalArgumentException(
          String.format(
              "Failed to parse signature %s of class %s with superclass %s and interfaces %s",
              signature, name, superName, Arrays.toString(interfaces)),
          e);
    }
  }

  private ClassSignatureParser(String signature, int expectedNumInterfaces) {
    this.index = 0;
    this.signature = signature;
    this.numInterfaces = 0;
    this.interfaceTypeParameters = new String[expectedNumInterfaces];
  }

  private void readFromSignature() {
    checkState(index == 0, "readFromSignature should only be called once");
    // Generic type signature for any bound types (E.g. <T:Ljava/util/String>)
    processTypeParameters();
    generics = signature.substring(0, index);

    superclassSignature = parseSuperClassTypeSignature();

    // Superclasses can be inner classes of other types, separated by a '.'
    while (signature.charAt(index) == '.') {
      index++;

      SuperclassSignature innerClassSignatureInformation = parseSuperClassTypeSignature();

      // Reconcile the class identifier. We include the typeParameters of the superclass in the
      // identifier definition
      // of the inner class. Therefore, retrieve that information from the previously obtained
      // superclassSignature, while obtaining the typeParameters from
      // innerClassSignatureInformation.
      superclassSignature =
          SuperclassSignature.create(
              superclassSignature.identifier()
                  + superclassSignature.typeParameters()
                  + "."
                  + innerClassSignatureInformation.identifier(),
              innerClassSignatureInformation.typeParameters());
    }

    checkState(
        signature.charAt(index) == ';',
        "Expected last \";\" of superclass type definition, but got \"%s\" instead",
        signature.charAt(index));
    index++;

    while (index < signature.length()) {
      skipPackageSpecifierAndTypeName();

      // Generic type specifier that we need to copy to our emulated interface definition
      if (signature.charAt(index) == '<') {
        int startIndexOfTypeParameter = index;
        processTypeParameters();
        interfaceTypeParameters[numInterfaces] =
            signature.substring(startIndexOfTypeParameter, index);
      } else {
        // This has no generic specifier, so leave the spot empty to make index traversal easier
        // when processing the array
        interfaceTypeParameters[numInterfaces] = "";
      }

      checkState(
          signature.charAt(index) == ';',
          "Expected last \";\" of interface type definition, but got \"%s\" instead",
          signature.charAt(index));
      index++;

      numInterfaces++;
    }
  }

  private SuperclassSignature parseSuperClassTypeSignature() {
    int startIndexOfName = index;
    skipPackageSpecifierAndTypeName();
    String name = signature.substring(startIndexOfName, index);

    int startIndexOfGenerics = index;
    processTypeParameters();
    String generics = signature.substring(startIndexOfGenerics, index);

    return SuperclassSignature.create(name, generics);
  }

  private void skipPackageSpecifierAndTypeName() {
    while (signature.charAt(index) != ';' && signature.charAt(index) != '<') {
      index++;
    }
  }

  private void processTypeParameters() {
    if (signature.charAt(index) == '<') {
      int stack = 0;
      do {
        index++;
        // Handle nested typeParameters bounds
        if (signature.charAt(index) == '<') {
          stack++;
        }
      } while (signature.charAt(index) != '>' || stack-- > 0);

      checkState(
          signature.charAt(index) == '>',
          "Expected last \">\" of the type parameter, but got \"%s\" instead",
          signature.charAt(index));
      index++;
    }
  }

  /**
   * Container object to hold signature information as specified by <a
   * href="https://docs.oracle.com/javase/specs/jvms/se12/html/jvms-4.html#jvms-ClassSignature">$4.7.9.1</a>
   * of the Java Virtual Machine Specification.
   */
  @AutoValue
  abstract static class ClassSignature {

    /**
     * Generic type parameters for this class, if any. It includes all type parameters and
     * surrounding angle brackets.
     *
     * @return Generic type parameters for this class, or an empty String if none exist
     */
    abstract String typeParameters();

    /**
     * Signature for the superclass.
     *
     * @return The signature for the superclass.
     */
    abstract SuperclassSignature superClassSignature();

    /**
     * List of generic type parameters for the interfaces the class implements. It includes all type
     * parameters for a single interface in 1 String.
     *
     * <p>For example, for the signature {@code
     * L__desugar__/java/util/List<TG;>;L__desugar__/java/util/Map<TK;TV;>;}. the
     * interfaceTypeParameters are {@code ["TG;", "TK;TV;"]}.
     *
     * @return List of type parameters for each interface the class implements.
     */
    abstract ImmutableList<String> interfaceTypeParameters();

    private static ClassSignature create(ClassSignatureParser classSignatureParser) {
      return new AutoValue_ClassSignatureParser_ClassSignature(
          classSignatureParser.generics,
          classSignatureParser.superclassSignature,
          ImmutableList.copyOf(classSignatureParser.interfaceTypeParameters));
    }
  }

  /**
   * Container object to hold signature information about the superclass. Name could include all
   * names and generic parameters of its superclasses. Generics will only contain the typeParameters
   * of the last class (in the case of inner classes).
   */
  @AutoValue
  abstract static class SuperclassSignature {

    /**
     * Name of the superclass. Includes packagespecifier.
     *
     * @return The full classname of this superclass.
     */
    abstract String identifier();

    /**
     * Generic type parameters for this class, if any. It includes all type parameters and
     * surrounding angle brackets.
     *
     * @return Generic type parameters for this class, or an empty String if none exist
     */
    abstract String typeParameters();

    private static SuperclassSignature create(String name, String generics) {
      return new AutoValue_ClassSignatureParser_SuperclassSignature(name, generics);
    }
  }
}
