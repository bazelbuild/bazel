// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.runfiles;


import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.annotation.processing.SupportedOptions;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.TypeElement;
import javax.tools.Diagnostic.Kind;

/** Processor for {@link AutoBazelRepository}. */
@SupportedAnnotationTypes("com.google.devtools.build.runfiles.AutoBazelRepository")
@SupportedOptions(AutoBazelRepositoryProcessor.BAZEL_REPOSITORY_OPTION)
public final class AutoBazelRepositoryProcessor extends AbstractProcessor {

  static final String BAZEL_REPOSITORY_OPTION = "bazel.repository";

  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latestSupported();
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    annotations.stream()
        .flatMap(element -> roundEnv.getElementsAnnotatedWith(element).stream())
        .map(element -> (TypeElement) element)
        .forEach(this::emitClass);
    return false;
  }

  private void emitClass(TypeElement annotatedClass) {
    // This option is always provided by the Java rule implementations.
    if (!processingEnv.getOptions().containsKey(BAZEL_REPOSITORY_OPTION)) {
      processingEnv
          .getMessager()
          .printMessage(
              Kind.ERROR,
              String.format(
                  "The %1$s annotation processor option is not set. To use this annotation"
                      + " processor, provide the canonical repository name of the current target as"
                      + " the value of the -A%1$s flag.",
                  BAZEL_REPOSITORY_OPTION),
              annotatedClass);
      return;
    }
    String repositoryName = processingEnv.getOptions().get(BAZEL_REPOSITORY_OPTION);
    if (repositoryName == null) {
      // javac translates '-Abazel.repository=' into a null value.
      // https://github.com/openjdk/jdk/blob/7a49c9baa1d4ad7df90e7ca626ec48ba76881822/src/jdk.compiler/share/classes/com/sun/tools/javac/processing/JavacProcessingEnvironment.java#L651
      repositoryName = "";
    }

    // For a nested class Outer.Middle.Inner, generate a class with simple name
    // AutoBazelRepository_Outer_Middle_Inner.
    // Note: There can be collisions when local classes are involved, but since the definition of a
    // class depends only on the containing Bazel target, this does not result in ambiguity.
    Deque<String> classNameSegments = new ArrayDeque<>();
    Element element = annotatedClass;
    while (element instanceof TypeElement) {
      classNameSegments.addFirst(element.getSimpleName().toString());
      element = element.getEnclosingElement();
    }
    classNameSegments.addFirst("AutoBazelRepository");
    String generatedClassSimpleName = String.join("_", classNameSegments);

    String generatedClassPackage =
        processingEnv.getElementUtils().getPackageOf(annotatedClass).getQualifiedName().toString();

    String generatedClassName =
        generatedClassPackage.isEmpty()
            ? generatedClassSimpleName
            : generatedClassPackage + "." + generatedClassSimpleName;

    try (PrintWriter out =
        new PrintWriter(
            processingEnv.getFiler().createSourceFile(generatedClassName).openWriter())) {
      
      if (!generatedClassPackage.isEmpty()) {
        // This annotation may exist on a class which is at the root package
        out.printf("package %s;\n", generatedClassPackage);
      }
      out.printf("\n");
      out.printf("class %s {\n", generatedClassSimpleName);
      out.printf("  /**\n");
      out.printf("   * The canonical name of the repository containing the Bazel target that\n");
      out.printf("   * compiled {@link %s}.\n", annotatedClass.getQualifiedName().toString());
      out.printf("   */\n");
      out.printf("  static final String NAME = \"%s\";\n", repositoryName);
      out.printf("\n");
      out.printf("  private %s() {}\n", generatedClassSimpleName);
      out.printf("}\n");
    } catch (IOException e) {
      processingEnv
          .getMessager()
          .printMessage(
              Kind.ERROR,
              String.format("Failed to generate %s: %s", generatedClassName, e.getMessage()),
              annotatedClass);
    }
  }
}
