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
package com.google.devtools.common.options.processor;

import java.lang.annotation.Annotation;
import java.util.Map;
import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.AnnotationValue;
import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;

/** Convenient utilities for dealing with the javax.lang.model types. */
public class ProcessorUtils {

  /** Return the AnnotationMirror for the annotation of the given type on the element provided. */
  static AnnotationMirror getAnnotation(
      Elements elementUtils,
      Types typeUtils,
      Element element,
      Class<? extends Annotation> annotation)
      throws OptionProcessorException {
    TypeElement annotationElement = elementUtils.getTypeElement(annotation.getCanonicalName());
    if (annotationElement == null) {
      // This can happen if the annotation is on the -processorpath but not on the -classpath.
      throw new OptionProcessorException(
          element, "Unable to find the type of annotation %s.", annotation);
    }
    TypeMirror annotationMirror = annotationElement.asType();

    for (AnnotationMirror annot : element.getAnnotationMirrors()) {
      if (typeUtils.isSameType(annot.getAnnotationType(), annotationMirror)) {
        return annot;
      }
    }
    // No annotation of this requested type found.
    throw new OptionProcessorException(
        element, "No annotation %s found for this element.", annotation);
  }

  /**
   * Returns the contents of a {@code Class}-typed field in an annotation.
   *
   * <p>Taken & adapted from AutoValueProcessor.java
   *
   * <p>This method is needed because directly reading the value of such a field from an
   * AnnotationMirror throws:
   *
   * <pre>
   * javax.lang.model.type.MirroredTypeException: Attempt to access Class object for TypeMirror Foo.
   * </pre>
   *
   * @param annotation The annotation to read from.
   * @param fieldName The name of the field to read, e.g. "exclude".
   * @return a set of fully-qualified names of classes appearing in 'fieldName' on 'annotation' on
   *     'element'.
   */
  static TypeElement getClassTypeFromAnnotationField(
      Elements elementUtils, AnnotationMirror annotation, String fieldName)
      throws OptionProcessorException {
    for (Map.Entry<? extends ExecutableElement, ? extends AnnotationValue> entry :
        elementUtils.getElementValuesWithDefaults(annotation).entrySet()) {
      if (entry.getKey().getSimpleName().contentEquals(fieldName)) {
        Object annotationField = entry.getValue().getValue();
        if (!(annotationField instanceof DeclaredType)) {
          throw new IllegalStateException(
              String.format(
                  "The fieldName provided should only apply to Class<> type annotation fields, "
                      + "but the field's value (%s) couldn't get cast to a DeclaredType",
                  entry));
        }
        String qualifiedName =
            ((TypeElement) ((DeclaredType) annotationField).asElement())
                .getQualifiedName()
                .toString();
        return elementUtils.getTypeElement(qualifiedName);
      }
    }
    // Annotation missing the requested field.
    throw new OptionProcessorException(
        null, "No member %s of the %s annotation found for element.", fieldName, annotation);
  }
}
