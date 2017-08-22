// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.common.options;

import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.Messager;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.annotation.processing.SupportedSourceVersion;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.TypeElement;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;
import javax.tools.Diagnostic;

/**
 * Annotation processor for {@link Option}.
 *
 * <p>The {@link OptionsParser} only accepts publicly declared options in {@link
 * OptionsBase}-inheriting classes, and there is no support for {@link Option} annotated fields
 * declared elsewhere or privately. Prevent such uses from compiling.
 */
@SupportedAnnotationTypes({"com.google.devtools.common.options.Option"})
@SupportedSourceVersion(SourceVersion.RELEASE_8)
public final class OptionProcessor extends AbstractProcessor {

  private Types typeUtils;
  private Elements elementUtils;
  private Messager messager;

  @Override
  public synchronized void init(ProcessingEnvironment processingEnv) {
    super.init(processingEnv);
    typeUtils = processingEnv.getTypeUtils();
    elementUtils = processingEnv.getElementUtils();
    messager = processingEnv.getMessager();
  }

  private static class OptionProcessorException extends Exception {
    private Element elementInError;

    OptionProcessorException(Element element, String message) {
      super(message);
      elementInError = element;
    }
  }

  /** Check that the Option variables only occur in OptionBase-inheriting classes. */
  private void checkInOptionBase(Element annotatedElement) throws OptionProcessorException {
    if (annotatedElement.getEnclosingElement().getKind() != ElementKind.CLASS) {
      throw new OptionProcessorException(annotatedElement, "The field should belong to a class.");
    }
    TypeMirror thisOptionClass = annotatedElement.getEnclosingElement().asType();
    TypeMirror optionsBase =
        elementUtils.getTypeElement("com.google.devtools.common.options.OptionsBase").asType();
    if (!typeUtils.isAssignable(thisOptionClass, optionsBase)) {
      throw new OptionProcessorException(
          annotatedElement,
          "@Option annotated fields can only be in classes that inherit from OptionsBase.");
    }
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    try {
      for (Element annotatedElement : roundEnv.getElementsAnnotatedWith(Option.class)) {
        // Only fields are annotated with Option, this should already be checked by the
        // @Target(ElementType.FIELD) annotation.

        checkInOptionBase(annotatedElement);
      }
    } catch (OptionProcessorException e) {
      error(e.elementInError, e.getMessage());
    }
    // Claim all Option annotated fields.
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
