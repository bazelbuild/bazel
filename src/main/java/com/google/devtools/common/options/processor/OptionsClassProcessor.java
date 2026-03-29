// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.common.options.processor;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsClass;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Locale;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.tools.Diagnostic;
import javax.tools.JavaFileObject;

/**
 * An annotation processor that generates an implementation class for options classes annotated with
 * {@link OptionsClass}.
 */
@SupportedAnnotationTypes("com.google.devtools.common.options.OptionsClass")
public final class OptionsClassProcessor extends AbstractProcessor {

  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latestSupported();
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    for (Element annotatedElement : roundEnv.getElementsAnnotatedWith(OptionsClass.class)) {
      if (annotatedElement.getKind() != ElementKind.CLASS) {
        continue;
      }
      TypeElement typeElement = (TypeElement) annotatedElement;
      generateWrapper(typeElement);
    }
    return false;
  }

  private void generateWrapper(TypeElement typeElement) {
    String packageName =
        processingEnv.getElementUtils().getPackageOf(typeElement).getQualifiedName().toString();
    String className = typeElement.getSimpleName().toString();
    if (!className.endsWith("Fields")) {
      processingEnv
          .getMessager()
          .printMessage(
              Diagnostic.Kind.ERROR,
              String.format(
                  "Class %s is annotated with @OptionsClass but its name does not end in"
                      + " 'Fields'",
                  className),
              typeElement);
      return;
    }
    String implClassName = className.substring(0, className.length() - "Fields".length());

    try {
      JavaFileObject jfo =
          processingEnv.getFiler().createSourceFile(packageName + "." + implClassName);
      try (PrintWriter out = new PrintWriter(jfo.openWriter())) {
        out.printf(
            """
            package %1$s;

            public class %2$s extends %3$s {
              public %2$s() {
                super();
              }
            """,
            packageName, implClassName, typeElement.getQualifiedName());

        // Find all elements annotated with @Option
        for (Element enclosed : typeElement.getEnclosedElements()) {
          if (enclosed.getAnnotation(Option.class) == null) {
            continue;
          }
          if (enclosed.getKind() != ElementKind.METHOD) {
            processingEnv
                .getMessager()
                .printMessage(
                    Diagnostic.Kind.ERROR,
                    "@Option must be on a method in @OptionsClass classes",
                    enclosed);
            continue;
          }

          ExecutableElement method = (ExecutableElement) enclosed;
          if (!method.getModifiers().contains(Modifier.PUBLIC)) {
            processingEnv
                .getMessager()
                .printMessage(Diagnostic.Kind.ERROR, "@Option method must be public", enclosed);
            continue;
          }

          String methodName = method.getSimpleName().toString();
          if (!methodName.startsWith("get")
              || methodName.length() < 4
              || !Character.isUpperCase(methodName.charAt(3))) {
            processingEnv
                .getMessager()
                .printMessage(
                    Diagnostic.Kind.ERROR,
                    "Annotated method name must start with 'get' followed by an uppercase"
                        + " letter",
                    enclosed);
            continue;
          }

          String fieldName =
              methodName.substring(3, 4).toLowerCase(Locale.ROOT) + methodName.substring(4);
          String fieldType = method.getReturnType().toString();
          String capitalizedFieldName = methodName.substring(3);

          out.printf(
              """
                private %1$s %2$s;

                @Override
                public %1$s get%3$s() {
                  return this.%2$s;
                }

                public void set%3$s(%1$s %2$s) {
                  this.%2$s = %2$s;
                }
              """,
              fieldType, fieldName, capitalizedFieldName);
        }

        out.println("}");
      }
    } catch (IOException e) {
      processingEnv
          .getMessager()
          .printMessage(
              Diagnostic.Kind.ERROR,
              "Failed to generate implementation for " + className + ": " + e.getMessage());
    }
  }
}
