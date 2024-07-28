// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.DeferredObjectCodecConstants.BUILDER_NAME;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.DeferredObjectCodecConstants.BUILDER_TYPE_NAME;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.DeferredObjectCodecConstants.CONSTRUCTOR_LOOKUP_NAME;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.DeferredObjectCodecConstants.makeSetterName;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.getErasure;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.resolveBaseArrayComponentType;

import com.google.devtools.build.lib.skyframe.serialization.ArrayProcessor;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.CodecHelpers;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.CodeBlock;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import com.squareup.javapoet.WildcardTypeName;
import java.lang.invoke.VarHandle;
import javax.annotation.Nullable;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.Name;
import javax.lang.model.type.ArrayType;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;

/**
 * Handles fields for {@link AutoCodec} general types.
 *
 * <p>This generates per-field specific code for {@link DeferredObjectCodec}s. Serialization occurs
 * by either using a getter or a {@link VarHandle} to retrieve the field value, then applying logic
 * that varies by type.
 *
 * <ul>
 *   <li>Primitives are serialized inline.
 *   <li>Arrays are serialized by delegating to {@link ArrayProcessor}.
 *   <li>Arbitrary objects are serialized by delegating to the {@link SerializationContext}.
 * </ul>
 *
 * <p>There's a similar division for deserialization, with the {@link
 * AsyncDeserializationContext#deserialize} handling aribtrary objects. Deserialized child values
 * are written to a builder object, generated in {@link DeferredObjectCodecGenerator} with all the
 * required fields and setters.
 */
abstract class DeferredObjectCodecFieldGenerators {

  static FieldGenerator create(
      Name name,
      TypeMirror parameterType,
      ClassName parentName,
      int hierarchyLevel,
      FieldAccessor accessor,
      ProcessingEnvironment env)
      throws SerializationProcessingException {
    TypeName typeName = getErasure(parameterType, env);
    switch (parameterType.getKind()) {
      case ARRAY:
        ArrayType arrayType = (ArrayType) parameterType;
        TypeMirror componentType = arrayType.getComponentType();
        TypeKind componentKind = componentType.getKind();
        if (componentKind.isPrimitive()) {
          return new PrimitiveArrayFieldGenerator(
              name, parameterType, typeName, parentName, componentKind, hierarchyLevel, accessor);
        }
        if (componentKind.equals(TypeKind.ARRAY)) {
          return new NestedArrayFieldGenerator(
              name,
              parameterType,
              typeName,
              parentName,
              hierarchyLevel,
              resolveBaseArrayComponentType(componentType).getKind(),
              accessor);
        }
        return new ObjectArrayFieldGenerator(
            name,
            parameterType,
            typeName,
            parentName,
            hierarchyLevel,
            getErasure(componentType, env),
            accessor);
      case BOOLEAN:
        return new BooleanFieldGenerator(
            name, parameterType, typeName, parentName, hierarchyLevel, accessor);
      case BYTE:
        return new ByteFieldGenerator(
            name, parameterType, typeName, parentName, hierarchyLevel, accessor);
      case CHAR:
        return new CharFieldGenerator(
            name, parameterType, typeName, parentName, hierarchyLevel, accessor);
      case DOUBLE:
        return new DoubleFieldGenerator(
            name, parameterType, typeName, parentName, hierarchyLevel, accessor);
      case FLOAT:
        return new FloatFieldGenerator(
            name, parameterType, typeName, parentName, hierarchyLevel, accessor);
      case INT:
        return new IntFieldGenerator(
            name, parameterType, typeName, parentName, hierarchyLevel, accessor);
      case LONG:
        return new LongFieldGenerator(
            name, parameterType, typeName, parentName, hierarchyLevel, accessor);
      case SHORT:
        return new ShortFieldGenerator(
            name, parameterType, typeName, parentName, hierarchyLevel, accessor);
      case TYPEVAR:
      case DECLARED:
        return new ObjectFieldGenerator(
            name, parameterType, typeName, parentName, hierarchyLevel, accessor);
      default:
        // There are other TypeKinds, for example, NONE, NULL, VOID and WILDCARD, (and more,
        // depending on the JDK version), but none are known to occur in code that defines the type
        // of a member variable.
        throw new SerializationProcessingException(
            env.getElementUtils().getTypeElement(parentName.canonicalName()),
            "%s had field %s having unexpected type %s",
            parentName,
            name,
            parameterType);
    }
  }

  /** Metadata needed to retrieve the field. */
  sealed interface FieldAccessor
      permits DeferredObjectCodecFieldGenerators.GetterName,
          DeferredObjectCodecFieldGenerators.FieldType {}

  /** Uses a getter method called {@code name} to retrieve the field. */
  record GetterName(String name) implements FieldAccessor {}

  /**
   * Uses a {@link VarHandle} to retrieve a field with type {@code typeName}.
   *
   * <p>The type is a parameter required to construct the {@link VarHandle}. It must be tracked
   * separately from the parameter field type because it can be a subclass. For example, an object
   * could be constructed using a {@code List} that is turned into an {@link ImmutableList} by the
   * constructor.
   */
  record FieldType(TypeName typeName) implements FieldAccessor {}

  private abstract static class VarHandleFieldGenerator extends FieldGenerator {
    private final FieldAccessor accessor;

    /**
     * Generates the per-field serialization code based on {@link VarHandle} and {@link
     * DeferredObjectCodec}.
     *
     * <p>As usual for {@link AutoCodec}, serializion is defined by an instantiator and this class
     * handles serialization of one of the <i>parameters</i> of the instantiator. This class
     * actually handles two different cases, retrieval using a {@link VarHandle} and retrieval using
     * getters, determined by the {@code accessor} implementation.
     *
     * @param accessor metadata about how to retrieve the field. The type {@link FieldAccessor} is
     *     sealed with two possible implementations. {@code accessor} is a {@link GetterName} when
     *     using a getter and contains the name of the getter. {@code accessor} is a {@link
     *     FieldType} when using a {@link VarHandle}, and it contains the exact field type, as
     *     defined in the class.
     */
    private VarHandleFieldGenerator(
        Name parameterName,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        int hierarchyLevel,
        FieldAccessor accessor) {
      super(parameterName, type, typeName, parentName, hierarchyLevel);
      this.accessor = checkNotNull(accessor);
    }

    @Override
    @Nullable
    final String getGetterName() {
      if (accessor instanceof GetterName getter) {
        return getter.name();
      }
      return null;
    }

    @Override
    final void generateHandleMember(TypeSpec.Builder classBuilder, MethodSpec.Builder constructor) {
      switch (accessor) {
        case GetterName unused -> {} // Using a getter. No VarHandle needed.
        case FieldType fieldType -> {
          classBuilder.addField(VarHandle.class, getHandleName(), Modifier.PRIVATE, Modifier.FINAL);
          constructor.addStatement(
              "this.$L = $L.findVarHandle($T.class, $S, $T.class)",
              getHandleName(),
              CONSTRUCTOR_LOOKUP_NAME + getHierarchyLevel(),
              getParentName(),
              getParameterName(),
              fieldType.typeName());
        }
      }
    }

    final CodeBlock generateAccessor() {
      var builder = CodeBlock.builder();
      switch (accessor) {
        case GetterName getterName -> builder.add("obj.$L()", getterName.name());
        case FieldType fieldType ->
            builder.add("($L) $L.get(obj)", fieldType.typeName(), getHandleName());
      }
      return builder.build();
    }
  }

  private static final class NestedArrayFieldGenerator extends VarHandleFieldGenerator {
    private final String processorName;

    private NestedArrayFieldGenerator(
        Name name,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        int hierarchyLevel,
        TypeKind baseComponentKind,
        FieldAccessor accessor) {
      super(name, type, typeName, parentName, hierarchyLevel, accessor);
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
          break;
        case DECLARED:
        case TYPEVAR:
          // See comments of `ArrayProcessor.OBJECT_ARRAY_PROCESSOR` to understand how it works for
          // any type of object array.
          this.processorName = "OBJECT_ARRAY_PROCESSOR";
          break;
        default:
          throw new IllegalStateException(
              "Unexpected base array component kind "
                  + baseComponentKind
                  + " for array field "
                  + name
                  + " of type "
                  + type);
      }
    }

    /**
     * The generated codec has a {@code Class<?>} member variable with this name storing the class
     * of the field.
     */
    private String getTypeMemberName() {
      return getNamePrefix() + "_type";
    }

    @Override
    void generateAdditionalMemberVariables(TypeSpec.Builder builder) {
      builder.addField(
          // Specifies Class<?> type.
          ParameterizedTypeName.get(
              ClassName.get(Class.class), WildcardTypeName.subtypeOf(Object.class)),
          getTypeMemberName(),
          Modifier.PRIVATE,
          Modifier.FINAL);
    }

    @Override
    void generateConstructorCode(MethodSpec.Builder constructor) {
      constructor.addStatement(
          "this.$L = $T.class.getDeclaredField(\"$L\").getType()",
          getTypeMemberName(),
          getParentName(),
          getParameterName());
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "$T.$L.serialize(context, codedOut, $L, $L)",
          ArrayProcessor.class,
          processorName,
          getTypeMemberName(),
          generateAccessor());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement(
          "$L.$L = ($T) $T.$L.deserialize(context, codedIn, $L)",
          BUILDER_NAME,
          getParameterName(),
          getTypeName(),
          ArrayProcessor.class,
          processorName,
          getTypeMemberName());
    }
  }

  private static class PrimitiveArrayFieldGenerator extends VarHandleFieldGenerator {
    private final TypeKind componentType;

    private PrimitiveArrayFieldGenerator(
        Name name,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        TypeKind componentType,
        int hierarchyLevel,
        FieldAccessor accessor) {
      super(name, type, typeName, parentName, hierarchyLevel, accessor);
      this.componentType = componentType;
    }

    private String getProcessorName() {
      return componentType + "_ARRAY_PROCESSOR";
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      String objName = getNamePrefix() + "_obj";
      serialize
          .addStatement("$T $L = $L", Object.class, objName, generateAccessor())
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
              "$L.$L = ($T) $T.$L.deserializeArrayData(codedIn, $L)",
              BUILDER_NAME,
              getParameterName(),
              getTypeName(),
              ArrayProcessor.class,
              getProcessorName(),
              lengthName)
          .endControlFlow();
    }
  }

  private static class ObjectArrayFieldGenerator extends VarHandleFieldGenerator {
    private final TypeName componentTypeName;

    private ObjectArrayFieldGenerator(
        Name name,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        int hierarchyLevel,
        TypeName componentTypeName,
        FieldAccessor accessor) {
      super(name, type, typeName, parentName, hierarchyLevel, accessor);
      this.componentTypeName = componentTypeName;
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      String arrName = getNamePrefix() + "_arr";
      serialize
          .addStatement("$T $L = $L", Object.class, arrName, generateAccessor())
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
      deserialize
          .addStatement("int $L = codedIn.readInt32()", lengthName)
          .beginControlFlow("if ($L > 0)", lengthName)
          .addStatement("$L--", lengthName)
          .addStatement(
              "$L.$L = new $T[$L]", BUILDER_NAME, getParameterName(), componentTypeName, lengthName)
          .addStatement(
              "$T.deserializeObjectArray(context, codedIn, $L.$L, $L)",
              ArrayProcessor.class,
              BUILDER_NAME,
              getParameterName(),
              lengthName)
          .endControlFlow();
    }
  }

  private static final class BooleanFieldGenerator extends VarHandleFieldGenerator {
    private BooleanFieldGenerator(
        Name name,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        int hierarchyLevel,
        FieldAccessor accessor) {
      super(name, type, typeName, parentName, hierarchyLevel, accessor);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement("codedOut.writeBoolNoTag((boolean) $L)", generateAccessor());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement("$L.$L = codedIn.readBool()", BUILDER_NAME, getParameterName());
    }
  }

  private static final class ByteFieldGenerator extends VarHandleFieldGenerator {
    private ByteFieldGenerator(
        Name name,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        int hierarchyLevel,
        FieldAccessor accessor) {
      super(name, type, typeName, parentName, hierarchyLevel, accessor);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement("codedOut.writeRawByte((byte) $L)", generateAccessor());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement("$L.$L = codedIn.readRawByte()", BUILDER_NAME, getParameterName());
    }
  }

  private static final class CharFieldGenerator extends VarHandleFieldGenerator {
    private CharFieldGenerator(
        Name name,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        int hierarchyLevel,
        FieldAccessor accessor) {
      super(name, type, typeName, parentName, hierarchyLevel, accessor);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "$T.writeChar(codedOut, (char) $L)", CodecHelpers.class, generateAccessor());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement(
          "$L.$L = $T.readChar(codedIn)", BUILDER_NAME, getParameterName(), CodecHelpers.class);
    }
  }

  private static final class DoubleFieldGenerator extends VarHandleFieldGenerator {
    private DoubleFieldGenerator(
        Name name,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        int hierarchyLevel,
        FieldAccessor accessor) {
      super(name, type, typeName, parentName, hierarchyLevel, accessor);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement("codedOut.writeDoubleNoTag((double) $L)", generateAccessor());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement("$L.$L = codedIn.readDouble()", BUILDER_NAME, getParameterName());
    }
  }

  private static final class FloatFieldGenerator extends VarHandleFieldGenerator {
    private FloatFieldGenerator(
        Name name,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        int hierarchyLevel,
        FieldAccessor accessor) {
      super(name, type, typeName, parentName, hierarchyLevel, accessor);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement("codedOut.writeFloatNoTag((float) $L)", generateAccessor());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement("$L.$L = codedIn.readFloat()", BUILDER_NAME, getParameterName());
    }
  }

  private static final class IntFieldGenerator extends VarHandleFieldGenerator {
    private IntFieldGenerator(
        Name name,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        int hierarchyLevel,
        FieldAccessor accessor) {
      super(name, type, typeName, parentName, hierarchyLevel, accessor);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement("codedOut.writeInt32NoTag((int) $L)", generateAccessor());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement("$L.$L = codedIn.readInt32()", BUILDER_NAME, getParameterName());
    }
  }

  private static final class LongFieldGenerator extends VarHandleFieldGenerator {
    private LongFieldGenerator(
        Name name,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        int hierarchyLevel,
        FieldAccessor accessor) {
      super(name, type, typeName, parentName, hierarchyLevel, accessor);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement("codedOut.writeInt64NoTag((long) $L)", generateAccessor());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement("$L.$L = codedIn.readInt64()", BUILDER_NAME, getParameterName());
    }
  }

  private static final class ShortFieldGenerator extends VarHandleFieldGenerator {
    private ShortFieldGenerator(
        Name name,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        int hierarchyLevel,
        FieldAccessor accessor) {
      super(name, type, typeName, parentName, hierarchyLevel, accessor);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement(
          "$T.writeShort(codedOut, (short) $L)", CodecHelpers.class, generateAccessor());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement(
          "$L.$L = $T.readShort(codedIn)", BUILDER_NAME, getParameterName(), CodecHelpers.class);
    }
  }

  private static final class ObjectFieldGenerator extends VarHandleFieldGenerator {
    private ObjectFieldGenerator(
        Name name,
        TypeMirror type,
        TypeName typeName,
        ClassName parentName,
        int hierarchyLevel,
        FieldAccessor accessor) {
      super(name, type, typeName, parentName, hierarchyLevel, accessor);
    }

    @Override
    void generateSerializeCode(MethodSpec.Builder serialize) {
      serialize.addStatement("context.serialize($L, codedOut)", generateAccessor());
    }

    @Override
    void generateDeserializeCode(MethodSpec.Builder deserialize) {
      deserialize.addStatement(
          "context.deserialize(codedIn, $L, $L::$L)",
          BUILDER_NAME,
          BUILDER_TYPE_NAME,
          makeSetterName(getParameterName()));
    }
  }
}
