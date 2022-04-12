// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.autocodec;

import static com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationProcessorUtil.getGeneratedName;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationProcessorUtil.sanitizeTypeParameter;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationProcessorUtil.writeGeneratedClassToFile;

import com.google.auto.service.AutoService;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.CodecScanningConstants;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationProcessorUtil.SerializationProcessingFailedException;
import com.squareup.javapoet.FieldSpec;
import com.squareup.javapoet.TypeSpec;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.Processor;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.tools.Diagnostic;

/** Javac annotation processor for static final fields that should be serialization constants. */
@AutoService(Processor.class)
public class SerializationConstantProcessor extends AbstractProcessor {
  private ProcessingEnvironment env;

  @Override
  public synchronized void init(ProcessingEnvironment processingEnv) {
    super.init(processingEnv);
    this.env = processingEnv;
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    try {
      processInternal(roundEnv);
    } catch (SerializationProcessingFailedException e) {
      // Reporting a message with ERROR kind will fail compilation.
      env.getMessager().printMessage(Diagnostic.Kind.ERROR, e.getMessage(), e.getElement());
    }
    return false;
  }

  private void processInternal(RoundEnvironment roundEnv)
      throws SerializationProcessingFailedException {
    for (Element element : roundEnv.getElementsAnnotatedWith(SerializationConstant.class)) {
      writeGeneratedClassToFile(
          element, buildRegisteredSingletonClass((VariableElement) element, env), env);
    }
  }

  private static final ImmutableList<Modifier> REQUIRED_SINGLETON_MODIFIERS =
      ImmutableList.of(Modifier.STATIC, Modifier.FINAL);

  static TypeSpec buildRegisteredSingletonClass(VariableElement symbol, ProcessingEnvironment env)
      throws SerializationProcessingFailedException {
    if (!symbol.getModifiers().containsAll(REQUIRED_SINGLETON_MODIFIERS)) {
      throw new SerializationProcessingFailedException(
          symbol,
          "Field must be static and final to be annotated with @SerializationConstant or"
              + " @AutoCodec");
    }
    return TypeSpec.classBuilder(
            getGeneratedName(symbol, CodecScanningConstants.REGISTERED_SINGLETON_SUFFIX))
        .addModifiers(Modifier.PUBLIC)
        .addSuperinterface(RegisteredSingletonDoNotUse.class)
        .addField(
            FieldSpec.builder(
                    Object.class,
                    CodecScanningConstants.REGISTERED_SINGLETON_INSTANCE_VAR_NAME,
                    Modifier.PUBLIC,
                    Modifier.STATIC,
                    Modifier.FINAL)
                .initializer(
                    "$T.$L",
                    sanitizeTypeParameter(symbol.getEnclosingElement().asType(), env),
                    symbol.getSimpleName())
                .build())
        .build();
  }

  @Override
  public Set<String> getSupportedAnnotationTypes() {
    return ImmutableSet.of(SerializationConstant.class.getCanonicalName());
  }

  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latestSupported(); // Supports all versions of Java.
  }
}
