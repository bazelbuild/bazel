/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.optimize.gson;

/**
 * Constants used in the classes of the GSON library.
 *
 * @author Lars Vandenbergh
 */
public class GsonClassConstants
{
    public static final String NAME_GSON_BUILDER = "com/google/gson/GsonBuilder";

    public static final String METHOD_NAME_ADD_DESERIALIZATION_EXCLUSION_STRATEGY   = "addDeserializationExclusionStrategy";
    public static final String METHOD_TYPE_ADD_DESERIALIZATION_EXCLUSION_STRATEGY   = "(Lcom/google/gson/ExclusionStrategy;)Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_ADD_SERIALIZATION_EXCLUSION_STRATEGY     = "addSerializationExclusionStrategy";
    public static final String METHOD_TYPE_ADD_SERIALIZATION_EXCLUSION_STRATEGY     = "(Lcom/google/gson/ExclusionStrategy;)Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_DISABLE_INNER_CLASS_SERIALIZATION        = "disableInnerClassSerialization";
    public static final String METHOD_TYPE_DISABLE_INNER_CLASS_SERIALIZATION        = "()Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_ENABLE_COMPLEX_MAP_KEY_SERIALIZATION     = "enableComplexMapKeySerialization";
    public static final String METHOD_TYPE_ENABLE_COMPLEX_MAP_KEY_SERIALIZATION     = "()Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_EXCLUDE_FIELDS_WITH_MODIFIERS            = "excludeFieldsWithModifiers";
    public static final String METHOD_TYPE_EXCLUDE_FIELDS_WITH_MODIFIERS            = "([I)Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_EXCLUDE_FIELDS_WITHOUT_EXPOSE_ANNOTATION = "excludeFieldsWithoutExposeAnnotation";
    public static final String METHOD_TYPE_EXLCUDE_FIELDS_WITHOUT_EXPOSE_ANNOTATION = "()Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_GENERATE_EXECUTABLE_JSON                 = "generateNonExecutableJson";
    public static final String METHOD_TYPE_GENERATE_EXECUTABLE_JSON                 = "()Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_REGISTER_TYPE_ADAPTER                    = "registerTypeAdapter";
    public static final String METHOD_TYPE_REGISTER_TYPE_ADAPTER                    = "(Ljava/lang/reflect/Type;Ljava/lang/Object;)Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_REGISTER_TYPE_HIERARCHY_ADAPTER          = "registerTypeHierarchyAdapter";
    public static final String METHOD_TYPE_REGISTER_TYPE_HIERARCHY_ADAPTER          = "(Ljava/lang/Class;Ljava/lang/Object;)Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_SERIALIZE_NULLS                          = "serializeNulls";
    public static final String METHOD_TYPE_SERIALIZE_NULLS                          = "()Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_SERIALIZE_SPECIAL_FLOATING_POINT_VALUES  = "serializeSpecialFloatingPointValues";
    public static final String METHOD_TYPE_SERIALIZE_SPECIAL_FLOATING_POINT_VALUES  = "()Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_SET_EXCLUSION_STRATEGIES                 = "setExclusionStrategies";
    public static final String METHOD_TYPE_SET_EXCLUSION_STRATEGIES                 = "([Lcom/google/gson/ExclusionStrategy;)Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_SET_FIELD_NAMING_POLICY                  = "setFieldNamingPolicy";
    public static final String METHOD_TYPE_SET_FIELD_NAMING_POLICY                  = "(Lcom/google/gson/FieldNamingPolicy;)Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_SET_FIELD_NAMING_STRATEGY                = "setFieldNamingStrategy";
    public static final String METHOD_TYPE_SET_FIELD_NAMING_STRATEGY                = "(Lcom/google/gson/FieldNamingStrategy;)Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_SET_LONG_SERIALIZATION_POLICY            = "setLongSerializationPolicy";
    public static final String METHOD_TYPE_SET_LONG_SERIALIZATION_POLICY            = "(Lcom/google/gson/LongSerializationPolicy;)Lcom/google/gson/GsonBuilder;";
    public static final String METHOD_NAME_SET_VERSION                              = "setVersion";
    public static final String METHOD_TYPE_SET_VERSION                              = "(D)Lcom/google/gson/GsonBuilder;";

    public static final String NAME_GSON = "com/google/gson/Gson";

    public static final String FIELD_NAME_EXCLUDER              = "excluder";
    public static final String FIELD_TYPE_EXCLUDER              = "Lcom/google/gson/internal/Excluder;";
    public static final String FIELD_NAME_FIELD_NAMING_STRATEGY = "fieldNamingStrategy";
    public static final String FIELD_TYPE_FIELD_NAMING_STRATEGY = "Lcom/google/gson/FieldNamingStrategy;";
    public static final String FIELD_NAME_INSTANCE_CREATORS     = "instanceCreators";
    public static final String FIELD_TYPE_INSTANCE_CREATORS     = "Ljava/util/Map;";
    public static final String FIELD_NAME_TYPE_TOKEN_CACHE      = "typeTokenCache";
    public static final String FIELD_TYPE_TYPE_TOKEN_CACHE      = "Ljava/util/Map;";

    public static final String METHOD_NAME_GET_ADAPTER_CLASS      = "getAdapter";
    public static final String METHOD_TYPE_GET_ADAPTER_CLASS      = "(Ljava/lang/Class;)Lcom/google/gson/TypeAdapter;";
    public static final String METHOD_NAME_GET_ADAPTER_TYPE_TOKEN = "getAdapter";
    public static final String METHOD_TYPE_GET_ADAPTER_TYPE_TOKEN = "(Lcom/google/gson/reflect/TypeToken;)Lcom/google/gson/TypeAdapter;";

    public static final String METHOD_NAME_TO_JSON                        = "toJson";
    public static final String METHOD_TYPE_TO_JSON_OBJECT                 = "(Ljava/lang/Object;)Ljava/lang/String;";
    public static final String METHOD_TYPE_TO_JSON_OBJECT_TYPE            = "(Ljava/lang/Object;Ljava/lang/reflect/Type;)Ljava/lang/String;";
    public static final String METHOD_TYPE_TO_JSON_OBJECT_APPENDABLE      = "(Ljava/lang/Object;Ljava/lang/Appendable;)V";
    public static final String METHOD_TYPE_TO_JSON_OBJECT_TYPE_APPENDABLE = "(Ljava/lang/Object;Ljava/lang/reflect/Type;Ljava/lang/Appendable;)V";
    public static final String METHOD_TYPE_TO_JSON_OBJECT_TYPE_WRITER     = "(Ljava/lang/Object;Ljava/lang/reflect/Type;Lcom/google/gson/stream/JsonWriter;)V";
    public static final String METHOD_TYPE_TO_JSON_JSON_ELEMENT_WRITER    = "(Lcom/google/gson/JsonElement;Lcom/google/gson/stream/JsonWriter;)V";

    public static final String METHOD_NAME_TO_JSON_TREE                   = "toJsonTree";
    public static final String METHOD_TYPE_TO_JSON_TREE_OBJECT            = "(Ljava/lang/Object;)Lcom/google/gson/JsonElement;";
    public static final String METHOD_TYPE_TO_JSON_TREE_OBJECT_TYPE       = "(Ljava/lang/Object;Ljava/lang/reflect/Type;)Lcom/google/gson/JsonElement;";

    public static final String METHOD_NAME_FROM_JSON                    = "fromJson";
    public static final String METHOD_TYPE_FROM_JSON_STRING_CLASS       = "(Ljava/lang/String;Ljava/lang/Class;)Ljava/lang/Object;";
    public static final String METHOD_TYPE_FROM_JSON_STRING_TYPE        = "(Ljava/lang/String;Ljava/lang/reflect/Type;)Ljava/lang/Object;";
    public static final String METHOD_TYPE_FROM_JSON_READER_CLASS       = "(Ljava/io/Reader;Ljava/lang/Class;)Ljava/lang/Object;";
    public static final String METHOD_TYPE_FROM_JSON_READER_TYPE        = "(Ljava/io/Reader;Ljava/lang/reflect/Type;)Ljava/lang/Object;";
    public static final String METHOD_TYPE_FROM_JSON_JSON_READER_TYPE   = "(Lcom/google/gson/stream/JsonReader;Ljava/lang/reflect/Type;)Ljava/lang/Object;";

    public static final String NAME_TYPE_TOKEN          = "com/google/gson/reflect/TypeToken";
    public static final String METHOD_NAME_GET_TYPE     = "getType";
    public static final String METHOD_TYPE_GET_TYPE     = "()Ljava/lang/reflect/Type;";
    public static final String METHOD_NAME_GET_RAW_TYPE = "getRawType";
    public static final String METHOD_TYPE_GET_RAW_TYPE = "()Ljava/lang/Class;";

    public static final String NAME_EXCLUDER                         = "com/google/gson/internal/Excluder";
    public static final String FIELD_NAME_DESERIALIZATION_STRATEGIES = "deserializationStrategies";
    public static final String FIELD_TYPE_DESERIALIZATION_STRATEGIES = "Ljava/util/List;";
    public static final String FIELD_NAME_MODIFIERS                  = "modifiers";
    public static final String FIELD_TYPE_MODIFIERS                  = "I";
    public static final String FIELD_NAME_SERIALIZATION_STRATEGIES   = "serializationStrategies";
    public static final String FIELD_TYPE_SERIALIZATION_STRATEGIES   = "Ljava/util/List;";
    public static final String FIELD_NAME_VERSION                    = "version";
    public static final String FIELD_TYPE_VERSION                    = "D";

    public static final String NAME_INSTANCE_CREATOR = "com/google/gson/InstanceCreator";

    public static final String METHOD_NAME_CREATE_INSTANCE = "createInstance";
    public static final String METHOD_TYPE_CREATE_INSTANCE = "(Ljava/lang/reflect/Type;)Ljava/lang/Object;";

    public static final String NAME_TYPE_ADAPTER = "com/google/gson/TypeAdapter";
    public static final String METHOD_NAME_READ  = "read";
    public static final String METHOD_TYPE_READ  = "(Lcom/google/gson/stream/JsonReader;)Ljava/lang/Object;";
    public static final String METHOD_NAME_WRITE = "write";
    public static final String METHOD_TYPE_WRITE = "(Lcom/google/gson/stream/JsonWriter;Ljava/lang/Object;)V";

    public static final String NAME_FIELD_NAMING_POLICY = "com/google/gson/FieldNamingPolicy";
    public static final String FIELD_NAME_IDENTITY      = "IDENTITY";
    public static final String FIELD_TYPE_IDENTITY      = "Lcom/google/gson/FieldNamingPolicy;";

    public static final String NAME_JSON_READER                = "com/google/gson/stream/JsonReader";

    public static final String METHOD_NAME_READER_BEGIN_OBJECT = "beginObject";
    public static final String METHOD_TYPE_READER_BEGIN_OBJECT = "()V";
    public static final String METHOD_NAME_READER_END_OBJECT   = "endObject";
    public static final String METHOD_TYPE_READER_END_OBJECT   = "()V";
    public static final String METHOD_NAME_NEXT_STRING         = "nextString";
    public static final String METHOD_TYPE_NEXT_STRING         = "()Ljava/lang/String;";
    public static final String METHOD_NAME_NEXT_BOOLEAN        = "nextBoolean";
    public static final String METHOD_TYPE_NEXT_BOOLEAN        = "()Z";
    public static final String METHOD_NAME_NEXT_INTEGER        = "nextInt";
    public static final String METHOD_TYPE_NEXT_INTEGER        = "()I";
    public static final String METHOD_NAME_NEXT_NULL           = "nextNull";
    public static final String METHOD_TYPE_NEXT_NULL           = "()V";
    public static final String METHOD_NAME_SKIP_VALUE          = "skipValue";
    public static final String METHOD_TYPE_SKIP_VALUE          = "()V";

    public static final String NAME_JSON_WRITER                = "com/google/gson/stream/JsonWriter";

    public static final String METHOD_NAME_WRITER_BEGIN_OBJECT  = "beginObject";
    public static final String METHOD_TYPE_WRITER_BEGIN_OBJECT  = "()Lcom/google/gson/stream/JsonWriter;";
    public static final String METHOD_NAME_WRITER_END_OBJECT    = "endObject";
    public static final String METHOD_TYPE_WRITER_END_OBJECT    = "()Lcom/google/gson/stream/JsonWriter;";
    public static final String METHOD_NAME_HAS_NEXT             = "hasNext";
    public static final String METHOD_TYPE_HAS_NEXT             = "()Z";
    public static final String METHOD_NAME_PEEK                 = "peek";
    public static final String METHOD_TYPE_PEEK                 = "()Lcom/google/gson/stream/JsonToken;";
    public static final String METHOD_NAME_NULL_VALUE           = "nullValue";
    public static final String METHOD_TYPE_NULL_VALUE           = "()Lcom/google/gson/stream/JsonWriter;";
    public static final String METHOD_NAME_VALUE_BOOLEAN        = "value";
    public static final String METHOD_TYPE_VALUE_BOOLEAN        = "(Z)Lcom/google/gson/stream/JsonWriter;";
    public static final String METHOD_NAME_VALUE_BOOLEAN_OBJECT = "value";
    public static final String METHOD_TYPE_VALUE_BOOLEAN_OBJECT = "(Ljava/lang/Boolean;)Lcom/google/gson/stream/JsonWriter;";
    public static final String METHOD_NAME_VALUE_NUMBER         = "value";
    public static final String METHOD_TYPE_VALUE_NUMBER         = "(Ljava/lang/Number;)Lcom/google/gson/stream/JsonWriter;";
    public static final String METHOD_NAME_VALUE_STRING         = "value";
    public static final String METHOD_TYPE_NAME_VALUE_STRING    = "(Ljava/lang/String;)Lcom/google/gson/stream/JsonWriter;";

    public static final String NAME_JSON_SYNTAX_EXCEPTION = "com/google/gson/JsonSyntaxException";

    public static final String NAME_JSON_TOKEN    = "com/google/gson/stream/JsonToken";

    public static final String FIELD_NAME_NULL    = "NULL";
    public static final String FIELD_TYPE_NULL    = "Lcom/google/gson/stream/JsonToken;";
    public static final String FIELD_NAME_BOOLEAN = "BOOLEAN";
    public static final String FIELD_TYPE_BOOLEAN = "Lcom/google/gson/stream/JsonToken;";

    public static final String ANNOTATION_TYPE_EXPOSE          = "Lcom/google/gson/annotations/Expose;";
    public static final String ANNOTATION_TYPE_JSON_ADAPTER    = "Lcom/google/gson/annotations/JsonAdapter;";
    public static final String ANNOTATION_TYPE_SERIALIZED_NAME = "Lcom/google/gson/annotations/SerializedName;";
}
