// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.getErasure;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.resolveBaseArrayComponentType;

import com.google.devtools.build.lib.skyframe.serialization.ArrayProcessor;
import com.google.devtools.build.lib.skyframe.serialization.CodecHelpers;
import com.google.devtools.build.lib.unsafe.UnsafeProvider;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import com.squareup.javapoet.WildcardTypeName;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.ArrayType;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;

/** Handles fields for {@link AutoCodec} interned types. */
abstract class InterningObjectCodecFieldGenerators {
  /**
   * Creates a generator for the given variable.
   *
   * @param hierarchyLevel a variable could occur in either the class being serialized or in one of
   *     its ancestor classes. This is 0 for the class itself, 1 for its superclass, and so on. It
   *     is used to create unique names without risk of shadowing.
   */
  static FieldGenerator create(
      VariableElement variable, int hierarchyLevel, ProcessingEnvironment env)
      throws SerializationProcessingException {
    TypeMirror type = variable.asType();
    switch (type.getKind()) {
      case ARRAY:
        ArrayType arrayType = (ArrayType) type;
        TypeMirror componentType = arrayType.getComponentType();
        TypeKind componentKind = componentType.getKind();
        if (componentKind.isPrimitive()) {
          return new PrimitiveArrayFieldGenerator(variable, componentKind, hierarchyLevel);
        }
        if (componentKind.equals(TypeKind.ARRAY)) {
          return new NestedArrayFieldGenerator(
              variable, hierarchyLevel, resolveBaseArrayComponentType(componentType).getKind());
        }
        return new ObjectArrayFieldGenerator(
            variable, hierarchyLevel, getErasure(componentType, env));
      case BOOLEAN:
        return new BooleanFieldGenerator(variable, hierarchyLevel);
      case BYTE:
        return new ByteFieldGenerator(variable, hierarchyLevel);
      case CHAR:
        return new CharFieldGenerator(variable, hierarchyLevel);
      case DOUBLE:
        return new DoubleFieldGenerator(variable, hierarchyLevel);
      case FLOAT:
        return new FloatFieldGenerator(variable, hierarchyLevel);
      case INT:
        return new IntFieldGenerator(variable, hierarchyLevel);
      case LONG:
        return new LongFieldGenerator(variable, hierarchyLevel);
      case SHORT:
        return new ShortFieldGenerator(variable, hierarchyLevel);
      case TYPEVAR:
      case DECLARED:
        return new ObjectFieldGenerator(variable, hierarchyLevel);
      default:
        // There are other TypeKinds, for example, NONE, NULL, VOID and WILDCARD, (and more,
        // depending on the JDK version), but none are known to occur in code that defines the type
        // of a member variable.
        TypeElement parent = (TypeElement) variable.getEnclosingElement();
        throw new SerializationProcessingException(
            parent,
            "%s had field %s having unexpected type %s",
            parent.getQualifiedName(),
            variable.getSimpleName(),
            type);
    }
  }

  /** Implementation that uses field offsets as handles. */
  private abstract static class OffsetFieldGenerator extends FieldGenerator {
    private OffsetFieldGenerator(VariableElement variable, int hierarchyLevel) {
      super(variable, hierarchyLevel);
    }

    @Override
    final void generateHandleMember(TypeSpec.Builder classBuilder, MethodSpec.Builder constructor) {
      classBuilder.addField(long.class, getHandleName(), Modifier.PRIVATE, Modifier.FINAL);
      constructor.addStatement(
          "this.$L = $T.unsafe().objectFieldOffset($T.class.getDeclaredField(\"$L\"))",
          getHandleName(),
          UnsafeProvider.class,
          getParentName(),
          getParameterName());
    }
  }

  private static final class NestedArrayFieldGenerator extends OffsetFieldGenerator {
    private final String processorName;
    private final boolean isPrimitiveArray;

    private NestedArrayFieldGenerator(
        VariableElement variable, int hierarchyLevel, TypeKind baseComponentKind) {
      super(variable, hierarchyLevel);
      switch (baseComponentKind) {
        case BOOLEAN:
        case BYTE:
        case CHAR:
        case DOUBLE:
        case FLOAT:
        case INT:
        case LONG:
        case SHORT:
          this.processorName = baseComponentKind.name() + "_ARRAY_PROCESSOR";
          this.isPrimitiveArray = true;
          break;
        case DECLARED:
        case TYPEVAR:
          // See comments of `ArrayProcessor.OBJECT_ARRAY_PROCESSOR` to understand how it works for
          // any type of object array.
          this.processorName = "OBJECT_ARRAY_PROCESSOR";
          this.isPrimitiveArray = false;
          break;
        default:
          throw new IllegalStateException(
              "Unexpected base array component kind "
                  + baseComponentKind
                  + " for array field "
                  + variable);
      }
    }

    private String getTypeName() {
      return getNamePrefix() + "_type";
    }

    @Override
    void generateAdditionalMemberVariables(TypeSpec.Builder builder) {
      builder.addField(
          // Specifies Class<?> type.
          ParameterizedTypeName.get(
              ClassName.get(Class.class), WildcardTypeName.subtypeOf(Object.class)),
          getTypeName(),
          Modifier.PRIVATE,
          Modifier.FINAL);
    }

    @Override
    void generateConstructorCode(MethodSpec.Builder constructor) {
      constructor.addStatement(
          "this.$L = $T.class.getDeclaredField(\"$L\").getType()",
          getTypeName(),
          getParentName(),
          getParameterName());
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "$T.$L.serialize(context, codedOut, $L, $T.unsafe().getObject(obj, $L))",
          ArrayProcessor.class,
          processorName,
          getTypeName(),
          UnsafeProvider.class,
          getHandleName());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      if (isPrimitiveArray) {
        deserialize.addStatement(
            "$T.$L.deserialize(codedIn, $L, instance, $L)",
            ArrayProcessor.class,
            processorName,
            getTypeName(),
            getHandleName());
      } else {
        deserialize.addStatement(
            "$T.$L.deserialize(context, codedIn, $L, instance, $L)",
            ArrayProcessor.class,
            processorName,
            getTypeName(),
            getHandleName());
      }
    }
  }

  private static class PrimitiveArrayFieldGenerator extends OffsetFieldGenerator {
    private final TypeKind componentType;

    private PrimitiveArrayFieldGenerator(
        VariableElement variable, TypeKind componentType, int hierarchyLevel) {
      super(variable, hierarchyLevel);
      this.componentType = componentType;
    }

    private String getProcessorName() {
      return componentType + "_ARRAY_PROCESSOR";
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      String objName = getNamePrefix() + "_obj";
      serialize
          .addStatement(
              "$T $L = $T.unsafe().getObject(obj, $L)",
              Object.class,
              objName,
              UnsafeProvider.class,
              getHandleName())
          .beginControlFlow("if ($L == null)", objName)
          .addStatement("codedOut.writeInt32NoTag(0)")
          .nextControlFlow("else")
          .addStatement(
              "$T.$L.serializeArrayData(codedOut, $L)",
              ArrayProcessor.class,
              getProcessorName(),
              objName)
          .endControlFlow();
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      String lengthName = getNamePrefix() + "_length";
      deserialize
          .addStatement("int $L = codedIn.readInt32()", lengthName)
          .beginControlFlow("if ($L > 0)", lengthName)
          .addStatement("$L--", lengthName)
          .addStatement(
              "$T.unsafe().putObject(instance, $L, $T.$L.deserializeArrayData(codedIn, $L))",
              UnsafeProvider.class,
              getHandleName(),
              ArrayProcessor.class,
              getProcessorName(),
              lengthName)
          .endControlFlow();
    }
  }

  private static class ObjectArrayFieldGenerator extends OffsetFieldGenerator {
    private final TypeName componentTypeName;

    private ObjectArrayFieldGenerator(
        VariableElement variable, int hierarchyLevel, TypeName componentTypeName) {
      super(variable, hierarchyLevel);
      this.componentTypeName = componentTypeName;
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      String arrName = getNamePrefix() + "_arr";
      serialize
          .addStatement(
              "$T $L = $T.unsafe().getObject(obj, $L)",
              Object.class,
              arrName,
              UnsafeProvider.class,
              getHandleName())
          .beginControlFlow("if ($L == null)", arrName)
          .addStatement("codedOut.writeInt32NoTag(0)")
          .nextControlFlow("else")
          .addStatement(
              "$T.serializeObjectArray(context, codedOut, $L)", ArrayProcessor.class, arrName)
          .endControlFlow();
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      String lengthName = getNamePrefix() + "_length";
      String arrName = getNamePrefix() + "_arr";
      deserialize
          .addStatement("int $L = codedIn.readInt32()", lengthName)
          .beginControlFlow("if ($L > 0)", lengthName)
          .addStatement("$L--", lengthName)
          .addStatement(
              "$T[] $L = new $T[$L]", componentTypeName, arrName, componentTypeName, lengthName)
          .addStatement(
              "$T.unsafe().putObject(instance, $L, $L)",
              UnsafeProvider.class,
              getHandleName(),
              arrName)
          .addStatement(
              "$T.deserializeObjectArray(context, codedIn, $L, $L)",
              ArrayProcessor.class,
              arrName,
              lengthName)
          .endControlFlow();
    }
  }

  private static class BooleanFieldGenerator extends OffsetFieldGenerator {
    private BooleanFieldGenerator(VariableElement variable, int hierarchyLevel) {
      super(variable, hierarchyLevel);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "codedOut.writeBoolNoTag($T.unsafe().getBoolean(obj, $L))",
          UnsafeProvider.class,
          getHandleName());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement(
          "$T.unsafe().putBoolean(instance, $L, codedIn.readBool())",
          UnsafeProvider.class,
          getHandleName());
    }
  }

  private static class ByteFieldGenerator extends OffsetFieldGenerator {
    private ByteFieldGenerator(VariableElement variable, int hierarchyLevel) {
      super(variable, hierarchyLevel);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "codedOut.writeRawByte($T.unsafe().getByte(obj, $L))",
          UnsafeProvider.class,
          getHandleName());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement(
          "$T.unsafe().putByte(instance, $L, codedIn.readRawByte())",
          UnsafeProvider.class,
          getHandleName());
    }
  }

  private static class CharFieldGenerator extends OffsetFieldGenerator {
    private CharFieldGenerator(VariableElement variable, int hierarchyLevel) {
      super(variable, hierarchyLevel);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "$T.writeChar(codedOut, $T.unsafe().getChar(obj, $L))",
          CodecHelpers.class,
          UnsafeProvider.class,
          getHandleName());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement(
          "$T.unsafe().putChar(instance, $L, $T.readChar(codedIn))",
          UnsafeProvider.class,
          getHandleName(),
          CodecHelpers.class);
    }
  }

  private static class DoubleFieldGenerator extends OffsetFieldGenerator {
    private DoubleFieldGenerator(VariableElement variable, int hierarchyLevel) {
      super(variable, hierarchyLevel);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "codedOut.writeDoubleNoTag($T.unsafe().getDouble(obj, $L))",
          UnsafeProvider.class,
          getHandleName());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement(
          "$T.unsafe().putDouble(instance, $L, codedIn.readDouble())",
          UnsafeProvider.class,
          getHandleName());
    }
  }

  private static class FloatFieldGenerator extends OffsetFieldGenerator {
    private FloatFieldGenerator(VariableElement variable, int hierarchyLevel) {
      super(variable, hierarchyLevel);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "codedOut.writeFloatNoTag($T.unsafe().getFloat(obj, $L))",
          UnsafeProvider.class,
          getHandleName());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement(
          "$T.unsafe().putFloat(instance, $L, codedIn.readFloat())",
          UnsafeProvider.class,
          getHandleName());
    }
  }

  private static class IntFieldGenerator extends OffsetFieldGenerator {
    private IntFieldGenerator(VariableElement variable, int hierarchyLevel) {
      super(variable, hierarchyLevel);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "codedOut.writeInt32NoTag($T.unsafe().getInt(obj, $L))",
          UnsafeProvider.class,
          getHandleName());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement(
          "$T.unsafe().putInt(instance, $L, codedIn.readInt32())",
          UnsafeProvider.class,
          getHandleName());
    }
  }

  private static class LongFieldGenerator extends OffsetFieldGenerator {
    private LongFieldGenerator(VariableElement variable, int hierarchyLevel) {
      super(variable, hierarchyLevel);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "codedOut.writeInt64NoTag($T.unsafe().getLong(obj, $L))",
          UnsafeProvider.class,
          getHandleName());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement(
          "$T.unsafe().putLong(instance, $L, codedIn.readInt64())",
          UnsafeProvider.class,
          getHandleName());
    }
  }

  private static class ShortFieldGenerator extends OffsetFieldGenerator {
    private ShortFieldGenerator(VariableElement variable, int hierarchyLevel) {
      super(variable, hierarchyLevel);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "$T.writeShort(codedOut, $T.unsafe().getShort(obj, $L))",
          CodecHelpers.class,
          UnsafeProvider.class,
          getHandleName());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement(
          "$T.unsafe().putShort(instance, $L, $T.readShort(codedIn))",
          UnsafeProvider.class,
          getHandleName(),
          CodecHelpers.class);
    }
  }

  private static class ObjectFieldGenerator extends OffsetFieldGenerator {
    private ObjectFieldGenerator(VariableElement variable, int hierarchyLevel) {
      super(variable, hierarchyLevel);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "context.serialize($T.unsafe().getObject(obj, $L), codedOut)",
          UnsafeProvider.class,
          getHandleName());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement("context.deserialize(codedIn, instance, $L)", getHandleName());
    }
  }
}
