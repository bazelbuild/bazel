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
import com.google.gson.stream.*;

import java.io.IOException;

/**
 * Template class for an optimized GSON type adapter.
 *
 * The implementation of the write() and read() methods need to be replaced
 * with injected byte code that invokes the generated toJson$xxx() and
 * fromJson$xxx() methods on the appropriate domain class.
 */
public class _OptimizedTypeAdapterImpl
extends      TypeAdapter
implements   _OptimizedTypeAdapter
{
    private Gson                 gson;
    private _OptimizedJsonReader optimizedJsonReader;
    private _OptimizedJsonWriter optimizedJsonWriter;


    /**
     * Creates a new _OptimizedTypeAdapterImpl.
     *
     * @param gson                the Gson context.
     * @param optimizedJsonReader the optimized reader used to read Json.
     * @param optimizedJsonWriter the optimized writer used to write Json.
     */
    public _OptimizedTypeAdapterImpl(Gson gson, _OptimizedJsonReader optimizedJsonReader, _OptimizedJsonWriter optimizedJsonWriter)
    {
        this.gson = gson;
        this.optimizedJsonReader = optimizedJsonReader;
        this.optimizedJsonWriter = optimizedJsonWriter;
    }


    // Implementations for TypeAdapter.

    @Override
    public void write(JsonWriter writer, Object value) throws IOException
    {
    }

    @Override
    public Object read(JsonReader reader) throws IOException
    {
        return null;
    }
}