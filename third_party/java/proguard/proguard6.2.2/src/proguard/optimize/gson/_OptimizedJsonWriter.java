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

import com.google.gson.stream.JsonWriter;

import java.io.IOException;

/*
 * Interface for writing Json fields and values using an internal index.
 * This allows injecting optimized Java code that writes out Json without
 * referring to Json field names and values using plain Strings.
 *
 * @author Lars Vandenbergh
 */
public interface _OptimizedJsonWriter
{
    /**
     * Writes the field name with the given internal index to the given Json writer.
     *
     * The original name of this method is "name".
     *
     * The name of this field has already been obfuscated because it is part
     * of an injected class.
     *
     * When renaming this field, the corresponding constant in
     * OptimizedClassConstants needs to be updated accordingly.
     *
     * @param jsonWriter   the Json writer to write to.
     * @param nameIndex    the internal index of the field name.
     * @throws IOException if the writing failed.
     */
    void b(JsonWriter jsonWriter, int nameIndex) throws IOException;

    /**
     * Writes the field value with the given internal index to the given Json writer.
     *
     * The original name of this method is "value".
     *
     * The name of this method has already been obfuscated because it is part
     * of an injected class.
     *
     * When renaming this method, the corresponding constant in
     * OptimizedClassConstants needs to be updated accordingly.
     *
     * @param jsonWriter   the Json writer to write to.
     * @param valueIndex   the internal index of the field value.
     * @throws IOException if the writing failed.
     */
    void c(JsonWriter jsonWriter, int valueIndex) throws IOException;
}
