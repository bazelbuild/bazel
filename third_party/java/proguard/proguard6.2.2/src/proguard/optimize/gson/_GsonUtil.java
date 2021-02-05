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

import com.google.gson.*;
import com.google.gson.internal.bind.ReflectiveTypeAdapterFactory;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.*;
import java.util.Map;

/**
 * This utility class is injected into the program class pool when the GSON
 * optimizations are applied. It contains the logic for picking the right
 * type adapter for a given type and value that needs to be serialized.
 *
 * The injected toJson() methods in the domain classes will use these utility
 * methods for serializing the fields of the domain class with the appropriate
 * type adapter.
 *
 * @author Lars Vandenbergh
 */
public final class _GsonUtil
{
    /**
     * Returns the appropriate type adapter for handling the given value with
     * the given declared type.
     *
     * @param gson         the Gson context that manages all registered type
     *                     adapters.
     * @param declaredType the type of the value to (de)serialize.
     * @param value        the value to (de)serialize.
     * @return             the type adapter for handling the given value and
     *                     declared type.
     */
    public static TypeAdapter getTypeAdapter(Gson gson, Class declaredType, Object value)
    {
        // If the runtime type is a sub type and there is a custom type adapter registered for
        // the declared type, that one should get precedence over the runtime type adapter if
        // the runtime type adapter is not custom.
        Type runtimeType = getRuntimeTypeIfMoreSpecific(declaredType, value);
        TypeAdapter runtimeTypeAdapter = gson.getAdapter(TypeToken.get(runtimeType));
        if (declaredType != runtimeType && !isCustomTypeAdapter(runtimeTypeAdapter))
        {
            TypeAdapter declaredTypeAdapter = gson.getAdapter(declaredType);
            if (isCustomTypeAdapter(declaredTypeAdapter))
            {
                return declaredTypeAdapter;
            }
        }

        // In all other cases the type adapter for the runtime type is used.
        return runtimeTypeAdapter;
    }


    /**
     * Returns the appropriate type adapter for handling the given value with
     * the given declared type token.
     *
     * @param gson              the Gson context that manages all registered type
     *                          adapters.
     * @param declaredTypeToken the declared type token of the value to (de)serialize.
     * @param value             the value to (de)serialize.
     * @return                  the type adapter for handling the given value and
     *                          declared type.
     */
    public static TypeAdapter getTypeAdapter(Gson gson, TypeToken declaredTypeToken, Object value)
    {
        // If the runtime type is a sub type and there is a custom type adapter registered for
        // the declared type, that one should get precedence over the runtime type adapter if
        // the runtime type adapter is not custom.
        Type declaredType = declaredTypeToken.getType();
        Type runtimeType = getRuntimeTypeIfMoreSpecific(declaredType, value);
        TypeAdapter runtimeTypeAdapter = gson.getAdapter(TypeToken.get(runtimeType));
        if (declaredType != runtimeType && !isCustomTypeAdapter(runtimeTypeAdapter))
        {
            TypeAdapter declaredTypeAdapter = gson.getAdapter(declaredTypeToken);
            if (isCustomTypeAdapter(declaredTypeAdapter))
            {
                return declaredTypeAdapter;
            }
        }

        // In all other cases the type adapter for the runtime type is used.
        return runtimeTypeAdapter;
    }


    /**
     * Finds a compatible runtime type if it is more specific
     */
    private static Type getRuntimeTypeIfMoreSpecific(Type type, Object value) {
        if (value != null
            && (type == Object.class || type instanceof TypeVariable<?> || type instanceof Class<?>)) {
            type = value.getClass();
        }
        return type;
    }

    /**
     * Determines whether a given type adapter is a custom type adapter, i.e.
     * a type adapter that is registered by the user of the Gson API and not
     * the GSON reflection based type adapter or the optimized type adapter
     * injected by DexGuard.
     */
    private static boolean isCustomTypeAdapter(TypeAdapter declaredTypeAdapter)
    {
        return !(declaredTypeAdapter instanceof _OptimizedTypeAdapter) &&
               !(declaredTypeAdapter instanceof ReflectiveTypeAdapterFactory.Adapter);
    }


    /**
     * Dumps the cached type adapter for each type for debugging purpose.
     */
    public static void dumpTypeTokenCache(String message, Map<TypeToken<?>, TypeAdapter<?>> typeTokenCache)
    {
        System.out.println(message);
        for (Map.Entry<TypeToken<?>, TypeAdapter<?>> typeTokenCacheEntry : typeTokenCache.entrySet())
        {
            System.out.println("    " + typeTokenCacheEntry.getKey() + " -> " + typeTokenCacheEntry.getValue());
        }
    }
}
