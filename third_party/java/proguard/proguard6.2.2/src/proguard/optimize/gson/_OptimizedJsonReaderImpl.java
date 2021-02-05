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

import com.google.gson.stream.*;

import java.io.IOException;
import java.util.*;

/**
 * This class is a template for an _OptimizedJsonReader implementation.
 * The data structure that contains the mapping between Json field names
 * and internal indices is empty and needs to be initialized using injected
 * byte code.
 *
 * @author Lars Vandenbergh
 */
public class _OptimizedJsonReaderImpl
implements   _OptimizedJsonReader
{
    /*
     * The original name of this field is "names".
     *
     * The name of this field has already been obfuscated because it is part
     * of an injected class.
     *
     * When renaming this field, the corresponding constant in
     * OptimizedClassConstants needs to be updated accordingly.
     */
    private static final Map<String, Integer> a = a();


    /*
     * Initializes the data structure containing the mapping between Json field
     * names and internal indices.
     *
     * The original name of this method is "initNames".
     *
     * The name of this method has already been obfuscated because it is part
     * of an injected class.
     *
     * When renaming this method, the corresponding constant in
     * OptimizedClassConstants needs to be updated accordingly.
     */
    private static Map<String, Integer> a()
    {
        return null;
    }

    // Implementations for _OptimizedJsonReader.

    @Override
    public int b(JsonReader jsonReader) throws IOException
    {
        String name = jsonReader.nextName();
        Integer fieldIndex = a.get(name);
        return fieldIndex == null ? -1 : fieldIndex;
    }

    @Override
    public int c(JsonReader jsonReader) throws IOException
    {
        String value = jsonReader.nextString();
        Integer valueIndex = a.get(value);
        return valueIndex == null ? -1 : valueIndex;
    }
}
