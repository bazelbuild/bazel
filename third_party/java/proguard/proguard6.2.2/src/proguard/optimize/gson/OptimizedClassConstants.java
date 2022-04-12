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

import proguard.classfile.util.ClassUtil;

/**
 * Constants used in the injected GSON optimization classes.
 *
 * @author Lars Vandenbergh
 */
public class OptimizedClassConstants
{
    // The class names contained in these strings will be mapped during
    // obfuscation by -adaptclassstrings.
    // We can't use .class.getName() for these template classes because it
    // would indirectly load Gson library classes which are not on the runtime
    // classpath of ProGuard.
    public static final String NAME_GSON_UTIL                      = "proguard/optimize/gson/_GsonUtil";
    public static final String NAME_OPTIMIZED_JSON_READER          = "proguard/optimize/gson/_OptimizedJsonReader";
    public static final String NAME_OPTIMIZED_JSON_READER_IMPL     = "proguard/optimize/gson/_OptimizedJsonReaderImpl";
    public static final String NAME_OPTIMIZED_JSON_WRITER          = "proguard/optimize/gson/_OptimizedJsonWriter";
    public static final String NAME_OPTIMIZED_JSON_WRITER_IMPL     = "proguard/optimize/gson/_OptimizedJsonWriterImpl";
    public static final String NAME_OPTIMIZED_TYPE_ADAPTER         = "proguard/optimize/gson/_OptimizedTypeAdapter";
    public static final String NAME_OPTIMIZED_TYPE_ADAPTER_FACTORY = "proguard/optimize/gson/_OptimizedTypeAdapterFactory";
    public static final String NAME_OPTIMIZED_TYPE_ADAPTER_IMPL    = "proguard/optimize/gson/_OptimizedTypeAdapterImpl";

    public static final String METHOD_NAME_INIT_NAMES_MAP      = "a";
    public static final String METHOD_TYPE_INIT_NAMES_MAP      = "()Ljava/util/Map;";
    public static final String METHOD_NAME_NEXT_FIELD_INDEX    = "b";
    public static final String METHOD_TYPE_NEXT_FIELD_INDEX    = "(Lcom/google/gson/stream/JsonReader;)I";
    public static final String METHOD_NAME_NEXT_VALUE_INDEX    = "c";
    public static final String METHOD_TYPE_NEXT_VALUE_INDEX    = "(Lcom/google/gson/stream/JsonReader;)I";

    public static final String METHOD_NAME_INIT_NAMES          = "a";
    public static final String METHOD_TYPE_INIT_NAMES          = "()[Ljava/lang/String;";
    public static final String METHOD_NAME_NAME                = "b";
    public static final String METHOD_TYPE_NAME                = "(Lcom/google/gson/stream/JsonWriter;I)V";
    public static final String METHOD_NAME_VALUE               = "c";
    public static final String METHOD_TYPE_VALUE               = "(Lcom/google/gson/stream/JsonWriter;I)V";

    public static final String FIELD_NAME_OPTIMIZED_JSON_READER_IMPL = "optimizedJsonReaderImpl";
    public static final String FIELD_TYPE_OPTIMIZED_JSON_READER_IMPL = ClassUtil.internalTypeFromClassName(NAME_OPTIMIZED_JSON_READER_IMPL);
    public static final String FIELD_NAME_OPTIMIZED_JSON_WRITER_IMPL = "optimizedJsonWriterImpl";
    public static final String FIELD_TYPE_OPTIMIZED_JSON_WRITER_IMPL = ClassUtil.internalTypeFromClassName(NAME_OPTIMIZED_JSON_WRITER_IMPL);
    public static final String METHOD_TYPE_INIT                      = "(Lcom/google/gson/Gson;" +
                                                                       ClassUtil.internalTypeFromClassName(NAME_OPTIMIZED_JSON_READER) +
                                                                       ClassUtil.internalTypeFromClassName(NAME_OPTIMIZED_JSON_WRITER) + ")V";
    public static final String METHOD_NAME_CREATE                    = "create";
    public static final String METHOD_TYPE_CREATE                    = "(Lcom/google/gson/Gson;Lcom/google/gson/reflect/TypeToken;)Lcom/google/gson/TypeAdapter;";


    public static final String TYPE_OPTIMIZED_TYPE_ADAPTER_IMPL = ClassUtil.internalTypeFromClassName(NAME_OPTIMIZED_TYPE_ADAPTER_IMPL);
    public static final String FIELD_NAME_GSON                  = "gson";
    public static final String FIELD_TYPE_GSON                  = "Lcom/google/gson/Gson;";
    public static final String FIELD_NAME_OPTIMIZED_JSON_READER = "optimizedJsonReader";
    public static final String FIELD_TYPE_OPTIMIZED_JSON_READER = ClassUtil.internalTypeFromClassName(NAME_OPTIMIZED_JSON_READER);
    public static final String FIELD_NAME_OPTIMIZED_JSON_WRITER = "optimizedJsonWriter";
    public static final String FIELD_TYPE_OPTIMIZED_JSON_WRITER = ClassUtil.internalTypeFromClassName(NAME_OPTIMIZED_JSON_WRITER);
    public static final String METHOD_NAME_READ                 = "read";
    public static final String METHOD_NAME_WRITE                = "write";

    public static final String FIELD_NAME_EXCLUDER              = "excluder";
    public static final String FIELD_TYPE_EXCLUDER              = "Lcom/google/gson/internal/Excluder;";
    public static final String FIELD_NAME_REQUIRE_EXPOSE        = "requireExpose";
    public static final String FIELD_TYPE_REQUIRE_EXPOSE        = "Z";

    public static final class ReadLocals
    {
        public static final int THIS                        = 0;
        public static final int JSON_READER                 = 1;
        public static final int VALUE                       = 2;
    }

    public static final class WriteLocals
    {
        public static final int THIS                        = 0;
        public static final int JSON_WRITER                 = 1;
        public static final int VALUE                       = 2;
    }


    public static final String METHOD_NAME_GET_TYPE_ADAPTER_CLASS      = "getTypeAdapter";
    public static final String METHOD_TYPE_GET_TYPE_ADAPTER_CLASS      = "(Lcom/google/gson/Gson;Ljava/lang/Class;Ljava/lang/Object;)Lcom/google/gson/TypeAdapter;";
    public static final String METHOD_NAME_GET_TYPE_ADAPTER_TYPE_TOKEN = "getTypeAdapter";
    public static final String METHOD_TYPE_GET_TYPE_ADAPTER_TYPE_TOKEN = "(Lcom/google/gson/Gson;Lcom/google/gson/reflect/TypeToken;Ljava/lang/Object;)Lcom/google/gson/TypeAdapter;";
    public static final String METHOD_NAME_DUMP_TYPE_TOKEN_CACHE       = "dumpTypeTokenCache";
    public static final String METHOD_TYPE_DUMP_TYPE_TOKEN_CACHE       = "(Ljava/lang/String;Ljava/util/Map;)V";

    public static final String METHOD_NAME_FROM_JSON       = "fromJson$";
    public static final String METHOD_TYPE_FROM_JSON       = "(Lcom/google/gson/Gson;Lcom/google/gson/stream/JsonReader;" +
                                                             ClassUtil.internalTypeFromClassName(NAME_OPTIMIZED_JSON_READER) + ")V";
    public static final String METHOD_NAME_FROM_JSON_FIELD = "fromJsonField$";
    public static final String METHOD_TYPE_FROM_JSON_FIELD = "(Lcom/google/gson/Gson;Lcom/google/gson/stream/JsonReader;I)V";

    public static final class FromJsonLocals
    {
        public static final int THIS                  = 0;
        public static final int GSON                  = 1;
        public static final int JSON_READER           = 2;
        public static final int OPTIMIZED_JSON_READER = 3;
        public static final int MAX_LOCALS            = 3;
    }

    public static final class FromJsonFieldLocals
    {
        public static final int THIS        = 0;
        public static final int GSON        = 1;
        public static final int JSON_READER = 2;
        public static final int FIELD_INDEX = 3;
    }

    public static final String METHOD_NAME_TO_JSON      = "toJson$";
    public static final String METHOD_TYPE_TO_JSON      = "(Lcom/google/gson/Gson;Lcom/google/gson/stream/JsonWriter;" +
                                                          ClassUtil.internalTypeFromClassName(NAME_OPTIMIZED_JSON_WRITER) + ")V";
    public static final String METHOD_NAME_TO_JSON_BODY = "toJsonBody$";
    public static final String METHOD_TYPE_TO_JSON_BODY = "(Lcom/google/gson/Gson;Lcom/google/gson/stream/JsonWriter;" +
                                                          ClassUtil.internalTypeFromClassName(NAME_OPTIMIZED_JSON_WRITER) + ")V";

    public static final class ToJsonLocals
    {
        public static final int THIS                  = 0;
        public static final int GSON                  = 1;
        public static final int JSON_WRITER           = 2;
        public static final int OPTIMIZED_JSON_WRITER = 3;
    }
}

