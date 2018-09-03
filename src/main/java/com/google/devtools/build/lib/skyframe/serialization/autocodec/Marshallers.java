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

import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationCodeGenerator.Context;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationCodeGenerator.Marshaller;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationCodeGenerator.PrimitiveValueSerializationCodeGenerator;
import com.google.devtools.build.lib.skyframe.serialization.strings.StringCodecs;
import com.squareup.javapoet.TypeName;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.TypeElement;
import javax.lang.model.type.ArrayType;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.PrimitiveType;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.type.TypeVariable;
import javax.lang.model.type.WildcardType;

/** Class containing all {@link Marshaller} instances. */
class Marshallers {
  private final ProcessingEnvironment env;

  Marshallers(ProcessingEnvironment env) {
    this.env = env;
  }

  void writeSerializationCode(Context context) {
    SerializationCodeGenerator generator = getMatchingCodeGenerator(context.type);
    boolean needsNullHandling = context.canBeNull() && generator != contextMarshaller;
    if (needsNullHandling) {
      context.builder.beginControlFlow("if ($L != null)", context.name);
      context.builder.addStatement("codedOut.writeBoolNoTag(true)");
    }
    generator.addSerializationCode(context);
    if (needsNullHandling) {
      context.builder.nextControlFlow("else");
      context.builder.addStatement("codedOut.writeBoolNoTag(false)");
      context.builder.endControlFlow();
    }
  }

  void writeDeserializationCode(Context context) {
    SerializationCodeGenerator generator = getMatchingCodeGenerator(context.type);
    boolean needsNullHandling = context.canBeNull() && generator != contextMarshaller;
    // If we have a generic or a wildcard parameter we need to erase it when we write the code out.
    TypeName contextTypeName = context.getTypeName();
    if (context.isDeclaredType() && !context.getDeclaredType().getTypeArguments().isEmpty()) {
      for (TypeMirror paramTypeMirror : context.getDeclaredType().getTypeArguments()) {
        if (isVariableOrWildcardType(paramTypeMirror)) {
          contextTypeName = TypeName.get(env.getTypeUtils().erasure(context.getDeclaredType()));
        }
      }
      // If we're just a generic or wildcard, get the erasure and use that.
    } else if (isVariableOrWildcardType(context.getTypeMirror())) {
      contextTypeName = TypeName.get(env.getTypeUtils().erasure(context.getTypeMirror()));
    }
    if (needsNullHandling) {
      context.builder.addStatement("$T $L = null", contextTypeName, context.name);
      context.builder.beginControlFlow("if (codedIn.readBool())");
    } else {
      context.builder.addStatement("$T $L", contextTypeName, context.name);
    }
    generator.addDeserializationCode(context);
    if (needsNullHandling) {
      context.builder.endControlFlow();
    }
  }

  private SerializationCodeGenerator getMatchingCodeGenerator(TypeMirror type) {
    if (type.getKind() == TypeKind.ARRAY) {
      return arrayCodeGenerator;
    }

    if (type instanceof PrimitiveType) {
      PrimitiveType primitiveType = (PrimitiveType) type;
      return primitiveGenerators
          .stream()
          .filter(generator -> generator.matches((PrimitiveType) type))
          .findFirst()
          .orElseThrow(() -> new IllegalArgumentException("No generator for: " + primitiveType));
    }

    // We're dealing with a generic.
    if (isVariableOrWildcardType(type)) {
      return contextMarshaller;
    }

    // TODO(cpeyser): Refactor primitive handling from AutoCodecProcessor.java
    if (!(type instanceof DeclaredType)) {
      throw new IllegalArgumentException(
          "Can only serialize primitive, array or declared fields, found " + type);
    }
    DeclaredType declaredType = (DeclaredType) type;

    return marshallers
        .stream()
        .filter(marshaller -> marshaller.matches(declaredType))
        .findFirst()
        .orElseThrow(
            () ->
                new IllegalArgumentException(
                    "No marshaller for: "
                        + ((TypeElement) declaredType.asElement()).getQualifiedName()));
  }

  private final SerializationCodeGenerator arrayCodeGenerator =
      new SerializationCodeGenerator() {
        @Override
        public void addSerializationCode(Context context) {
          String length = context.makeName("length");
          context.builder.addStatement("int $L = $L.length", length, context.name);
          context.builder.addStatement("codedOut.writeInt32NoTag($L)", length);
          Context repeated =
              context.with(
                  ((ArrayType) context.type).getComponentType(), context.makeName("repeated"));
          String indexName = context.makeName("i");
          context.builder.beginControlFlow(
              "for(int $L = 0; $L < $L; ++$L)", indexName, indexName, length, indexName);
          context.builder.addStatement(
              "$T $L = $L[$L]", repeated.getTypeName(), repeated.name, context.name, indexName);
          writeSerializationCode(repeated);
          context.builder.endControlFlow();
        }

        @Override
        public void addDeserializationCode(Context context) {
          Context repeated =
              context.with(
                  ((ArrayType) context.type).getComponentType(), context.makeName("repeated"));
          String lengthName = context.makeName("length");
          context.builder.addStatement("int $L = codedIn.readInt32()", lengthName);

          String resultName = context.makeName("result");
          context.builder.addStatement(
              "$T[] $L = new $T[$L]",
              repeated.getTypeName(),
              resultName,
              repeated.getTypeName(),
              lengthName);
          String indexName = context.makeName("i");
          context.builder.beginControlFlow(
              "for (int $L = 0; $L < $L; ++$L)", indexName, indexName, lengthName, indexName);
          writeDeserializationCode(repeated);
          context.builder.addStatement("$L[$L] = $L", resultName, indexName, repeated.name);
          context.builder.endControlFlow();
          context.builder.addStatement("$L = $L", context.name, resultName);
        }
      };

  private static final PrimitiveValueSerializationCodeGenerator INT_CODE_GENERATOR =
      new PrimitiveValueSerializationCodeGenerator() {
        @Override
        public boolean matches(PrimitiveType type) {
          return type.getKind() == TypeKind.INT;
        }

        @Override
        public void addSerializationCode(Context context) {
          context.builder.addStatement("codedOut.writeInt32NoTag($L)", context.name);
        }

        @Override
        public void addDeserializationCode(Context context) {
          context.builder.addStatement("$L = codedIn.readInt32()", context.name);
        }
      };

  private static final PrimitiveValueSerializationCodeGenerator LONG_CODE_GENERATOR =
      new PrimitiveValueSerializationCodeGenerator() {
        @Override
        public boolean matches(PrimitiveType type) {
          return type.getKind() == TypeKind.LONG;
        }

        @Override
        public void addSerializationCode(Context context) {
          context.builder.addStatement("codedOut.writeInt64NoTag($L)", context.name);
        }

        @Override
        public void addDeserializationCode(Context context) {
          context.builder.addStatement("$L = codedIn.readInt64()", context.name);
        }
      };

  private static final PrimitiveValueSerializationCodeGenerator BYTE_CODE_GENERATOR =
      new PrimitiveValueSerializationCodeGenerator() {
        @Override
        public boolean matches(PrimitiveType type) {
          return type.getKind() == TypeKind.BYTE;
        }

        @Override
        public void addSerializationCode(Context context) {
          context.builder.addStatement("codedOut.write($L)", context.name);
        }

        @Override
        public void addDeserializationCode(Context context) {
          context.builder.addStatement("$L = codedIn.readRawByte()", context.name);
        }
      };

  private static final PrimitiveValueSerializationCodeGenerator BOOLEAN_CODE_GENERATOR =
      new PrimitiveValueSerializationCodeGenerator() {
        @Override
        public boolean matches(PrimitiveType type) {
          return type.getKind() == TypeKind.BOOLEAN;
        }

        @Override
        public void addSerializationCode(Context context) {
          context.builder.addStatement("codedOut.writeBoolNoTag($L)", context.name);
        }

        @Override
        public void addDeserializationCode(Context context) {
          context.builder.addStatement("$L = codedIn.readBool()", context.name);
        }
      };

  private static final PrimitiveValueSerializationCodeGenerator DOUBLE_CODE_GENERATOR =
      new PrimitiveValueSerializationCodeGenerator() {
        @Override
        public boolean matches(PrimitiveType type) {
          return type.getKind() == TypeKind.DOUBLE;
        }

        @Override
        public void addSerializationCode(Context context) {
          context.builder.addStatement("codedOut.writeDoubleNoTag($L)", context.name);
        }

        @Override
        public void addDeserializationCode(Context context) {
          context.builder.addStatement("$L = codedIn.readDouble()", context.name);
        }
      };

  private final Marshaller charSequenceMarshaller =
      new Marshaller() {
        @Override
        public boolean matches(DeclaredType type) {
          return matchesType(type, CharSequence.class);
        }

        @Override
        public void addSerializationCode(Context context) {
          context.builder.addStatement(
              "$T.asciiOptimized().serialize(context, $L.toString(), codedOut)",
              StringCodecs.class,
              context.name);
        }

        @Override
        public void addDeserializationCode(Context context) {
          context.builder.addStatement(
              "$L = $T.asciiOptimized().deserialize(context, codedIn)",
              context.name,
              StringCodecs.class);
        }
      };

  private final Marshaller supplierMarshaller =
      new Marshaller() {
        @Override
        public boolean matches(DeclaredType type) {
          return matchesErased(type, Supplier.class);
        }

        @Override
        public void addSerializationCode(Context context) {
          DeclaredType suppliedType =
              (DeclaredType) context.getDeclaredType().getTypeArguments().get(0);
          writeSerializationCode(context.with(suppliedType, context.name + ".get()"));
        }

        @Override
        public void addDeserializationCode(Context context) {
          DeclaredType suppliedType =
              (DeclaredType) context.getDeclaredType().getTypeArguments().get(0);
          String suppliedName = context.makeName("supplied");
          writeDeserializationCode(context.with(suppliedType, suppliedName));
          String suppliedFinalName = context.makeName("suppliedFinal");
          context.builder.addStatement(
              "final $T $L = $L", suppliedType, suppliedFinalName, suppliedName);
          context.builder.addStatement("$L = () -> $L", context.name, suppliedFinalName);
        }
      };

  /** Delegates marshalling back to the context. */
  private final Marshaller contextMarshaller =
      new Marshaller() {
        @Override
        public boolean matches(DeclaredType unusedType) {
          return true;
        }

        @Override
        public void addSerializationCode(Context context) {
          context.builder.addStatement("context.serialize($L, codedOut)", context.name);
        }

        @Override
        public void addDeserializationCode(Context context) {
          context.builder.addStatement("$L = context.deserialize(codedIn)", context.name);
        }
      };

  private final ImmutableList<PrimitiveValueSerializationCodeGenerator> primitiveGenerators =
      ImmutableList.of(
          INT_CODE_GENERATOR,
          LONG_CODE_GENERATOR,
          BYTE_CODE_GENERATOR,
          BOOLEAN_CODE_GENERATOR,
          DOUBLE_CODE_GENERATOR);

  private final ImmutableList<Marshaller> marshallers =
      ImmutableList.of(
          charSequenceMarshaller,
          supplierMarshaller,
          contextMarshaller);

  /** True when {@code type} has the same type as {@code clazz}. */
  private boolean matchesType(TypeMirror type, Class<?> clazz) {
    return env.getTypeUtils().isSameType(type, getType(clazz));
  }

  /** True when erasure of {@code type} matches erasure of {@code clazz}. */
  private boolean matchesErased(TypeMirror type, Class<?> clazz) {
    return env.getTypeUtils()
        .isSameType(env.getTypeUtils().erasure(type), env.getTypeUtils().erasure(getType(clazz)));
  }

  /** Returns the TypeMirror corresponding to {@code clazz}. */
  private TypeMirror getType(Class<?> clazz) {
    return AutoCodecUtil.getType(clazz, env);
  }

  static boolean isVariableOrWildcardType(TypeMirror type) {
    return type instanceof TypeVariable || type instanceof WildcardType;
  }
}
