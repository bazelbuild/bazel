// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.getTypeMirror;
import static java.util.Arrays.stream;

import com.google.common.collect.ImmutableList;
import java.util.Optional;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.type.MirroredTypeException;
import javax.lang.model.type.MirroredTypesException;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;

/** A record storing the data in {@link AutoCodec}, used at annotation processing time. */
record AutoCodecAnnotation(
    boolean checkClassExplicitlyAllowed,
    ImmutableList<? extends TypeMirror> explicitlyAllowClass,
    Optional<TypeMirror> deserializedInterface) {
  static AutoCodecAnnotation of(AutoCodec annotation, ProcessingEnvironment env) {
    return new AutoCodecAnnotation(
        annotation.checkClassExplicitlyAllowed(),
        getExplicitlyAllowClass(annotation, env),
        getDeserializedInterface(annotation, env));
  }

  private static ImmutableList<? extends TypeMirror> getExplicitlyAllowClass(
      AutoCodec annotation, ProcessingEnvironment env) {
    try {
      return stream(annotation.explicitlyAllowClass())
          .map(clazz -> getTypeMirror(clazz, env))
          .collect(toImmutableList());
    } catch (MirroredTypesException e) {
      return ImmutableList.copyOf(e.getTypeMirrors());
    }
  }

  private static Optional<TypeMirror> getDeserializedInterface(
      AutoCodec annotation, ProcessingEnvironment env) {
    TypeMirror rawTypeMirror;
    try {
      rawTypeMirror = getTypeMirror(annotation.deserializedInterface(), env);
    } catch (MirroredTypeException e) {
      rawTypeMirror = e.getTypeMirror();
    }
    if (rawTypeMirror.getKind() == TypeKind.VOID) {
      return Optional.empty();
    }
    return Optional.of(rawTypeMirror);
  }
}
