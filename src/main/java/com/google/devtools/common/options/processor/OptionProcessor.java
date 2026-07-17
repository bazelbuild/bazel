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

import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.Messager;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.TypeElement;
import javax.tools.Diagnostic;

/**
 * Annotation processor for {@link Option}.
 *
 * <p>Checks the following invariants about {@link Option}-annotated fields ("options"):
 *
 * <ul>
 *   <li>The {@link OptionsParser} only accepts options in {@link OptionsBase}-inheriting classes
 *   <li>All options must be declared publicly and be neither static nor final.
 *   <li>All options that must be used on the command line must have sensible names without
 *       whitespace or other confusing characters, such as equal signs.
 *   <li>The type of the option must match the converter that will convert the unparsed string value
 *       into the option type. For options that do not specify a converter, check that there is a
 *       valid match in the {@link Converters#DEFAULT_CONVERTERS} list.
 *   <li>Options must list valid combinations of tags and documentation categories.
 *   <li>Expansion options and options with implicit requirements cannot expand in more than one
 *       way, how multiple expansions would interact is not defined and should not be necessary.
 *   <li>Multiple options must not declare default value (see {@link
 *       #MULTIPLE_OPTIONS_DEFAULT_VALUE_EXCEPTIONS} for exceptions).
 * </ul>
 *
 * <p>These properties can be relied upon at runtime without additional checks.
 */
@SupportedAnnotationTypes({"com.google.devtools.common.options.Option"})
public final class OptionProcessor extends AbstractProcessor {

  private Messager messager;

  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latestSupported();
  }

  @Override
  public synchronized void init(ProcessingEnvironment processingEnv) {
    super.init(processingEnv);
    messager = processingEnv.getMessager();
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    for (Element annotatedElement : roundEnv.getElementsAnnotatedWith(Option.class)) {
      if (annotatedElement.getKind() != ElementKind.FIELD) {
        continue;
      }
      error(annotatedElement, "Field options not supported anymore. Use method options instead.");
    }
    return false;
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
