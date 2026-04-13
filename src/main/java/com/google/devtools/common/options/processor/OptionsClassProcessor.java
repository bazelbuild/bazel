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
import java.util.ArrayList;
import java.util.List;
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
    String className =
        typeElement.getQualifiedName().toString().substring(packageName.length() + 1);
    String implClassName = className.replace('.', '_') + "Impl";

    record OptionInfo(String fieldType, String capitalizedFieldName, boolean hasSetterInBase) {}
    List<OptionInfo> options = new ArrayList<>();

    // First pass: collect option info
    for (Element member : processingEnv.getElementUtils().getAllMembers(typeElement)) {
      if (member.getAnnotation(Option.class) == null) {
        continue;
      }
      if (member.getKind() != ElementKind.METHOD) {
        processingEnv
            .getMessager()
            .printMessage(
                Diagnostic.Kind.ERROR,
                "@Option must be on a method in @OptionsClass classes",
                member);
        continue;
      }

      ExecutableElement method = (ExecutableElement) member;
      if (!method.getModifiers().contains(Modifier.ABSTRACT)) {
        processingEnv
            .getMessager()
            .printMessage(Diagnostic.Kind.ERROR, "@Option method must be abstract", member);
        continue;
      }
      if (!method.getModifiers().contains(Modifier.PUBLIC)) {
        processingEnv
            .getMessager()
            .printMessage(Diagnostic.Kind.ERROR, "@Option method must be public", member);
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
                "Annotated method name must start with 'get' followed by an uppercase letter",
                member);
        continue;
      }

      String fieldType = method.getReturnType().toString();
      String capitalizedFieldName = methodName.substring("get".length());

      ExecutableElement setter = null;
      String setterName = "set" + capitalizedFieldName;
      for (Element e : processingEnv.getElementUtils().getAllMembers(typeElement)) {
        if (e.getKind() == ElementKind.METHOD && e.getSimpleName().contentEquals(setterName)) {
          setter = (ExecutableElement) e;
          break;
        }
      }

      if (setter != null) {
        if (!setter.getModifiers().contains(Modifier.ABSTRACT)) {
          processingEnv
              .getMessager()
              .printMessage(Diagnostic.Kind.ERROR, "Setter must be abstract", setter);
          continue;
        }

        if (setter.getParameters().size() != 1) {
          processingEnv
              .getMessager()
              .printMessage(Diagnostic.Kind.ERROR, "Setter must have exactly one argument", setter);
          continue;
        }

        if (!processingEnv
            .getTypeUtils()
            .isSameType(setter.getParameters().get(0).asType(), method.getReturnType())) {
          processingEnv
              .getMessager()
              .printMessage(
                  Diagnostic.Kind.ERROR,
                  String.format(
                      "Setter argument type must be same as getter return type (%s)", fieldType),
                  setter);
          continue;
        }
      }

      options.add(new OptionInfo(fieldType, capitalizedFieldName, setter != null));
    }

    try {
      // Generate the Impl class
      JavaFileObject implJfo =
          processingEnv.getFiler().createSourceFile(packageName + "." + implClassName);
      try (PrintWriter out = new PrintWriter(implJfo.openWriter())) {
        out.printf(
            """
            package %1$s;

            public class %2$s extends %3$s {
              public %2$s() {
                super();
              }

              @Override
              @SuppressWarnings("unchecked")
              public Class<? extends %3$s> getOptionsClass() {
                return (Class<? extends %3$s>) %3$s.class;
              }
            """,
            packageName, implClassName, typeElement.getQualifiedName());

        for (OptionInfo option : options) {
          String fieldName =
              option.capitalizedFieldName.substring(0, 1).toLowerCase(Locale.ROOT)
                  + option.capitalizedFieldName.substring(1);
          out.printf(
              """
                private %1$s %2$s;

                @Override
                public %1$s get%3$s() {
                  return this.%2$s;
                }

                %4$s
                public void set%3$s(%1$s %2$s) {
                  this.%2$s = %2$s;
                }
              """,
              option.fieldType,
              fieldName,
              option.capitalizedFieldName,
              option.hasSetterInBase ? "@Override" : "");
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
