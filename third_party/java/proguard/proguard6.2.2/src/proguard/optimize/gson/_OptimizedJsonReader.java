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

/**
 * Interface for reading Json fields and values using an internal index.
 * This allows injecting optimized Java code that reads and interprets Json
 * without referring to Json field names and values using plain Strings.
 *
 * @author Lars Vandenbergh
 */
public interface _OptimizedJsonReader
{
    /**
     * Reads the internal index of the next Json field from the given Json
     * reader.
     *
     * The original name of this method is "nextFieldIndex".
     *
     * The name of this method has already been obfuscated because it is part
     * of an injected class.
     *
     * When renaming this field, the corresponding constant in
     * OptimizedClassConstants needs to be updated accordingly.
     *
     * @param jsonReader   the Json reader to read from.
     * @return             the internal index of the read field value.
     * @throws IOException if the reading failed.
     */
    int b(JsonReader jsonReader) throws IOException;

    /**
     * Reads the internal index of the next Json value from the given Json
     * reader.
     *
     * The original name of this method is "nextValueIndex".
     *
     * The name of this method has already been obfuscated because it is part
     * of an injected class.
     *
     * When renaming this method, the corresponding constant in
     * OptimizedClassConstants needs to be updated accordingly.
     *
     * @param jsonReader   the Json reader to read from.
     * @return             the internal index of the read field value.
     * @throws IOException if the reading failed.
     */
    int c(JsonReader jsonReader) throws IOException;
}
