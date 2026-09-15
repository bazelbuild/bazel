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
 * This class keeps track of which parameters of the GsonBuilder are being
 * utilized in the code.
 *
 * @author Lars Vandenbergh
 */
public class GsonRuntimeSettings
{
    public boolean setVersion;
    public boolean excludeFieldsWithModifiers;
    public boolean generateNonExecutableJson;
    public boolean excludeFieldsWithoutExposeAnnotation;
    public boolean serializeNulls;

    // This setting is taken care of by the built-in MapTypeAdapterFactory
    // of Gson.
    //    public boolean enableComplexMapKeySerialization;

    public boolean disableInnerClassSerialization;
    public boolean setLongSerializationPolicy;
    public boolean setFieldNamingPolicy;
    public boolean setFieldNamingStrategy;
    public boolean setExclusionStrategies;
    public boolean addSerializationExclusionStrategy;
    public boolean addDeserializationExclusionStrategy;

    // These settings on the builder are taken care of by the JsonWriter and
    // JsonReader and don't affect our optimizations.
    //    public boolean setPrettyPrinting;
    //    public boolean setLenient;
    //    public boolean disableHtmlEscaping;

    // This setting is taken care of by the built-in DateTypeAdapters of Gson.
    //    public boolean setDateFormat;

    // These type adapters come before the _OptimizedTypeAdapterFactory we
    // inject.
    //    public boolean registerTypeAdapter;
    //    public boolean registerTypeAdapterFactory;
    //    public boolean registerTypeHierarchyAdapter;

    public boolean serializeSpecialFloatingPointValues;
}
