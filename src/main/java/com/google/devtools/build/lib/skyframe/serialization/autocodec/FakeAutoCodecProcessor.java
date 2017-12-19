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

package com.google.devtools.build.lib.skyframe.serialization.autocodec;

import com.google.auto.service.AutoService;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import java.io.IOException;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.Processor;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.tools.Diagnostic;

/**
 * A fake annotation processor for bootstrapping.
 *
 * <p>When bootstrapping, annotation processors inside the Bazel source tree are not available which
 * means that if there are dependencies on generated code, bootstrapping will fail. This processor
 * generates stub code to pass the initial boostrapping phase.
 */
@AutoService(Processor.class)
public class FakeAutoCodecProcessor extends AbstractProcessor {
  private ProcessingEnvironment env; // Captured from `init` method.

  @Override
  public Set<String> getSupportedAnnotationTypes() {
    return ImmutableSet.of(AutoCodecUtil.ANNOTATION.getCanonicalName());
  }

  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latestSupported(); // Supports all versions of Java.
  }

  @Override
  public synchronized void init(ProcessingEnvironment processingEnv) {
    super.init(processingEnv);
    this.env = processingEnv;
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    for (Element element : roundEnv.getElementsAnnotatedWith(AutoCodecUtil.ANNOTATION)) {
      TypeElement encodedType = (TypeElement) element;
      TypeSpec.Builder codecClassBuilder =
          TypeSpec.classBuilder(AutoCodecUtil.getCodecName(encodedType))
              .addSuperinterface(
                  ParameterizedTypeName.get(
                      ClassName.get(ObjectCodec.class), TypeName.get(encodedType.asType())));
      codecClassBuilder.addMethod(
          MethodSpec.methodBuilder("getEncodedClass")
              .addModifiers(Modifier.PUBLIC)
              .addAnnotation(Override.class)
              .returns(
                  ParameterizedTypeName.get(
                      ClassName.get(Class.class), TypeName.get(encodedType.asType())))
              .addStatement("throw new RuntimeException(\"Shouldn't be called.\")")
              .build());
      codecClassBuilder.addMethod(
          MethodSpec.methodBuilder("serialize")
              .addModifiers(Modifier.PUBLIC)
              .addParameter(TypeName.get(encodedType.asType()), "input")
              .addParameter(CodedOutputStream.class, "codedOut")
              .addAnnotation(Override.class)
              .addStatement("throw new RuntimeException(\"Shouldn't be called.\")")
              .build());
      codecClassBuilder.addMethod(
          MethodSpec.methodBuilder("deserialize")
              .addModifiers(Modifier.PUBLIC)
              .returns(TypeName.get(encodedType.asType()))
              .addParameter(CodedInputStream.class, "codedIn")
              .addAnnotation(Override.class)
              .addStatement("throw new RuntimeException(\"Shouldn't be called.\")")
              .build());
      String packageName =
          env.getElementUtils().getPackageOf(encodedType).getQualifiedName().toString();
      try {
        JavaFile file = JavaFile.builder(packageName, codecClassBuilder.build()).build();
        file.writeTo(env.getFiler());
      } catch (IOException e) {
        env.getMessager()
            .printMessage(
                Diagnostic.Kind.ERROR, "Failed to generate output file: " + e.getMessage());
      }
    }
    return true;
  }
}
