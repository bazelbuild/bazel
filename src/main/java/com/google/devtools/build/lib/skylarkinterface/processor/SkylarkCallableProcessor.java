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

package com.google.devtools.build.lib.skylarkinterface.processor;

import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.Messager;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.annotation.processing.SupportedSourceVersion;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.tools.Diagnostic;

/**
 * Annotation processor for {@link SkylarkCallable}.
 *
 * <p>Checks the following invariants about {@link SkylarkCallable}-annotated methods:
 * <ul>
 * <li>The method must be public.</li>
 * <li>The number of method parameters much match the number of annotation-declared parameters.</li>
 * <li>If structField=true, there must be zero arguments.</li>
 * </ul>
 *
 * <p>These properties can be relied upon at runtime without additional checks.
 */
@SupportedAnnotationTypes({"com.google.devtools.build.lib.skylarkinterface.SkylarkCallable"})
@SupportedSourceVersion(SourceVersion.RELEASE_8)
public final class SkylarkCallableProcessor extends AbstractProcessor {

  private Messager messager;

  @Override
  public synchronized void init(ProcessingEnvironment processingEnv) {
    super.init(processingEnv);
    messager = processingEnv.getMessager();
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    for (Element element : roundEnv.getElementsAnnotatedWith(SkylarkCallable.class)) {
      // Only methods are annotated with SkylarkCallable. This is verified by the
      // @Target(ElementType.METHOD) annotation.
      ExecutableElement methodElement = (ExecutableElement) element;
      SkylarkCallable annotation = methodElement.getAnnotation(SkylarkCallable.class);

      if (!methodElement.getModifiers().contains(Modifier.PUBLIC)) {
        error(methodElement, "@SkylarkCallable annotated methods must be public.");
      }
      if (annotation.parameters().length > 0 || annotation.mandatoryPositionals() >= 0) {
         int numDeclaredArgs = annotation.parameters().length
             + Math.max(0, annotation.mandatoryPositionals());
        if (methodElement.getParameters().size() != numDeclaredArgs) {
          error(methodElement, String.format(
              "@SkylarkCallable annotated method has %d parameters, but annotation declared %d.",
              methodElement.getParameters().size(), numDeclaredArgs));
        }
      }
      if (annotation.structField()) {
        if (!methodElement.getParameters().isEmpty()) {
          error(methodElement,
              "@SkylarkCallable annotated methods with structField=true must have zero arguments.");
        }
      }
    }
    return true;
  }

  /**
   * Prints an error message & fails the compilation.
   *
   * @param e The element which has caused the error. Can be null
   * @param msg The error message
   */
  public void error(Element e, String msg) {
    messager.printMessage(Diagnostic.Kind.ERROR, msg, e);
  }
}
