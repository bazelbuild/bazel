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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.Marshaller.Context;
import com.google.devtools.build.lib.skyframe.serialization.strings.StringCodecs;
import com.google.protobuf.ProtocolMessageEnum;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.ElementKind;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.TypeMirror;

/** Class containing all {@link Marshaller} instances. */
class Marshallers {
  private final ProcessingEnvironment env;

  Marshallers(ProcessingEnvironment env) {
    this.env = env;
  }

  void writeSerializationCode(Context context) {
    context.builder.beginControlFlow("if ($L != null)", context.name);
    context.builder.addStatement("codedOut.writeBoolNoTag(true)");
    getMatchingMarshaller(context.type).addSerializationCode(context);
    context.builder.nextControlFlow("else");
    context.builder.addStatement("codedOut.writeBoolNoTag(false)");
    context.builder.endControlFlow();
  }

  void writeDeserializationCode(Context context) {
    context.builder.addStatement("$T $L = null", context.getTypeName(), context.name);
    context.builder.beginControlFlow("if (codedIn.readBool())");
    getMatchingMarshaller(context.type).addDeserializationCode(context);
    context.builder.endControlFlow();
  }

  private Marshaller getMatchingMarshaller(DeclaredType type) {
    return marshallers.stream().filter(marshaller -> marshaller.matches(type)).findFirst().get();
  }

  private static final Marshaller CODEC_MARSHALLER =
      new Marshaller() {
        @Override
        public boolean matches(DeclaredType type) {
          // CODEC is the final fallback for all Marshallers so this returns true.
          return true;
        }

        @Override
        public void addSerializationCode(Context context) {
          context.builder.addStatement(
              "$T.CODEC.serialize($L, codedOut)", context.getTypeName(), context.name);
        }

        @Override
        public void addDeserializationCode(Context context) {
          context.builder.addStatement(
              "$L = $T.CODEC.deserialize(codedIn)", context.name, context.getTypeName());
        }
      };

  private final Marshaller enumMarshaller =
      new Marshaller() {
        @Override
        public boolean matches(DeclaredType type) {
          return env.getTypeUtils().asElement(type).getKind() == ElementKind.ENUM;
        }

        @Override
        public void addSerializationCode(Context context) {
          if (isProtoEnum(context.type)) {
            context.builder.addStatement("codedOut.writeInt32NoTag($L.getNumber())", context.name);
          } else {
            context.builder.addStatement("codedOut.writeInt32NoTag($L.ordinal())", context.name);
          }
        }

        @Override
        public void addDeserializationCode(Context context) {
          if (isProtoEnum(context.type)) {
            context.builder.addStatement(
                "$L = $T.forNumber(codedIn.readInt32())", context.name, context.getTypeName());
          } else {
            // TODO(shahan): memoize this expensive call to values().
            context.builder.addStatement(
                "$L = $T.values()[codedIn.readInt32()]", context.name, context.getTypeName());
          }
        }

        private boolean isProtoEnum(DeclaredType type) {
          return env.getTypeUtils()
              .isSubtype(
                  type,
                  env.getElementUtils()
                      .getTypeElement(ProtocolMessageEnum.class.getCanonicalName())
                      .asType());
        }
      };

  private final Marshaller stringMarshaller =
      new Marshaller() {
        @Override
        public boolean matches(DeclaredType type) {
          return matchesType(type, String.class);
        }

        @Override
        public void addSerializationCode(Context context) {
          context.builder.addStatement(
              "$T.asciiOptimized().serialize($L, codedOut)", StringCodecs.class, context.name);
        }

        @Override
        public void addDeserializationCode(Context context) {
          context.builder.addStatement(
              "$L = $T.asciiOptimized().deserialize(codedIn)", context.name, StringCodecs.class);
        }
      };

  private final Marshaller mapEntryMarshaller =
      new Marshaller() {
        @Override
        public boolean matches(DeclaredType type) {
          return matchesErased(type, Map.Entry.class);
        }

        @Override
        public void addSerializationCode(Context context) {
          DeclaredType keyType = (DeclaredType) context.type.getTypeArguments().get(0);
          writeSerializationCode(context.with(keyType, context.name + ".getKey()"));
          DeclaredType valueType = (DeclaredType) context.type.getTypeArguments().get(1);
          writeSerializationCode(context.with(valueType, context.name + ".getValue()"));
        }

        @Override
        public void addDeserializationCode(Context context) {
          DeclaredType keyType = (DeclaredType) context.type.getTypeArguments().get(0);
          String keyName = context.makeName("key");
          writeDeserializationCode(context.with(keyType, keyName));
          DeclaredType valueType = (DeclaredType) context.type.getTypeArguments().get(1);
          String valueName = context.makeName("value");
          writeDeserializationCode(context.with(valueType, valueName));
          context.builder.addStatement(
              "$L = $T.immutableEntry($L, $L)", context.name, Maps.class, keyName, valueName);
        }
      };

  private final Marshaller listMarshaller =
      new Marshaller() {
        @Override
        public boolean matches(DeclaredType type) {
          // TODO(shahan): List is more general than ImmutableList. Consider whether these
          // two should have distinct marshallers.
          return matchesErased(type, List.class) || matchesErased(type, ImmutableList.class);
        }

        @Override
        public void addSerializationCode(Context context) {
          // Writes the target count to the stream so deserialization knows when to stop.
          context.builder.addStatement("codedOut.writeInt32NoTag($L.size())", context.name);
          Context repeated =
              context.with(
                  (DeclaredType) context.type.getTypeArguments().get(0),
                  context.makeName("repeated"));
          context.builder.beginControlFlow(
              "for ($T $L : $L)", repeated.getTypeName(), repeated.name, context.name);
          writeSerializationCode(repeated);
          context.builder.endControlFlow();
        }

        @Override
        public void addDeserializationCode(Context context) {
          Context repeated =
              context.with(
                  (DeclaredType) context.type.getTypeArguments().get(0),
                  context.makeName("repeated"));
          String builderName = context.makeName("builder");
          context.builder.addStatement(
              "$T<$T> $L = new $T<>()",
              ImmutableList.Builder.class,
              repeated.getTypeName(),
              builderName,
              ImmutableList.Builder.class);
          String lengthName = context.makeName("length");
          context.builder.addStatement("int $L = codedIn.readInt32()", lengthName);
          String indexName = context.makeName("i");
          context.builder.beginControlFlow(
              "for (int $L = 0; $L < $L; ++$L)", indexName, indexName, lengthName, indexName);
          writeDeserializationCode(repeated);
          context.builder.addStatement("$L.add($L)", builderName, repeated.name);
          context.builder.endControlFlow();
          context.builder.addStatement("$L = $L.build()", context.name, builderName);
        }
      };

  private final Marshaller immutableSortedSetMarshaller =
      new Marshaller() {
        @Override
        public boolean matches(DeclaredType type) {
          return matchesErased(type, ImmutableSortedSet.class);
        }

        @Override
        public void addSerializationCode(Context context) {
          listMarshaller.addSerializationCode(context);
        }

        @Override
        public void addDeserializationCode(Context context) {
          Context repeated =
              context.with(
                  (DeclaredType) context.type.getTypeArguments().get(0),
                  context.makeName("repeated"));
          String builderName = context.makeName("builder");
          context.builder.addStatement(
              "$T<$T> $L = new $T<>($T.naturalOrder())",
              ImmutableSortedSet.Builder.class,
              repeated.getTypeName(),
              builderName,
              ImmutableSortedSet.Builder.class,
              Comparator.class);
          String lengthName = context.makeName("length");
          context.builder.addStatement("int $L = codedIn.readInt32()", lengthName);
          String indexName = context.makeName("i");
          context.builder.beginControlFlow(
              "for (int $L = 0; $L < $L; ++$L)", indexName, indexName, lengthName, indexName);
          writeDeserializationCode(repeated);
          context.builder.addStatement("$L.add($L)", builderName, repeated.name);
          context.builder.endControlFlow();
          context.builder.addStatement("$L = $L.build()", context.name, builderName);
        }
      };

  private final Marshaller mapMarshaller =
      new Marshaller() {
        @Override
        public boolean matches(DeclaredType type) {
          // TODO(shahan): since Map is a bit more general than ImmutableSortedMap, consider
          // splitting these.
          return matchesErased(type, Map.class) || matchesErased(type, ImmutableSortedMap.class);
        }

        @Override
        public void addSerializationCode(Context context) {
          context.builder.addStatement("codedOut.writeInt32NoTag($L.size())", context.name);
          String entryName = context.makeName("entry");
          Context key =
              context.with(
                  (DeclaredType) context.type.getTypeArguments().get(0), entryName + ".getKey()");
          Context value =
              context.with(
                  (DeclaredType) context.type.getTypeArguments().get(1), entryName + ".getValue()");
          context.builder.beginControlFlow(
              "for ($T<$T, $T> $L : $L.entrySet())",
              Map.Entry.class,
              key.getTypeName(),
              value.getTypeName(),
              entryName,
              context.name);
          writeSerializationCode(key);
          writeSerializationCode(value);
          context.builder.endControlFlow();
        }

        @Override
        public void addDeserializationCode(Context context) {
          Context key =
              context.with(
                  (DeclaredType) context.type.getTypeArguments().get(0), context.makeName("key"));
          Context value =
              context.with(
                  (DeclaredType) context.type.getTypeArguments().get(1), context.makeName("value"));
          String builderName = context.makeName("builder");
          context.builder.addStatement(
              "$T<$T, $T> $L = new $T<>($T.naturalOrder())",
              ImmutableSortedMap.Builder.class,
              key.getTypeName(),
              value.getTypeName(),
              builderName,
              ImmutableSortedMap.Builder.class,
              Comparator.class);
          String lengthName = context.makeName("length");
          context.builder.addStatement("int $L = codedIn.readInt32()", lengthName);
          String indexName = context.makeName("i");
          context.builder.beginControlFlow(
              "for (int $L = 0; $L < $L; ++$L)", indexName, indexName, lengthName, indexName);
          writeDeserializationCode(key);
          writeDeserializationCode(value);
          context.builder.addStatement("$L.put($L, $L)", builderName, key.name, value.name);
          context.builder.endControlFlow();
          context.builder.addStatement("$L = $L.build()", context.name, builderName);
        }
      };

  private final ImmutableList<Marshaller> marshallers =
      ImmutableList.of(
          enumMarshaller,
          stringMarshaller,
          mapEntryMarshaller,
          listMarshaller,
          immutableSortedSetMarshaller,
          mapMarshaller,
          CODEC_MARSHALLER);

  /** True when {@code type} has the same type as {@code clazz}. */
  private boolean matchesType(TypeMirror type, Class<?> clazz) {
    return env.getTypeUtils()
        .isSameType(
            type, env.getElementUtils().getTypeElement((clazz.getCanonicalName())).asType());
  }

  /** True when erasure of {@code type} matches erasure of {@code clazz}. */
  private boolean matchesErased(TypeMirror type, Class<?> clazz) {
    return env.getTypeUtils()
        .isSameType(
            env.getTypeUtils().erasure(type),
            env.getTypeUtils()
                .erasure(
                    env.getElementUtils().getTypeElement((clazz.getCanonicalName())).asType()));
  }
}
