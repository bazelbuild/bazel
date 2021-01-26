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

import proguard.classfile.*;
import proguard.classfile.editor.*;

/**
 * Interface for injecting optimized code for deserializing a field of a class
 * from Json.
 *
 * @author Lars Vandenbergh
 */
interface InlineDeserializer
{
    /**
     * Indicates whether the deserializer can inject optimized code given which
     * GSON builder invocations are utilized in the program code.
     *
     * @param  gsonRuntimeSettings tracks the GSON parameters that are utilized
     *                             in the code.
     * @return true if and only if the deserializer can inject optimized code.
     */
    boolean canDeserialize(GsonRuntimeSettings gsonRuntimeSettings);

    /**
     * Appends optimized code for deserializing the given field of the given class
     * using the given code attribute editor and instruction sequence builder.
     *
     * The current locals are:
     * 0 this (the domain object)
     * 1 gson
     * 2 jsonReader
     * 3 fieldIndex
     *
     * @param programClass         The domain class containing the field to
     *                             deserialize.
     * @param programField         The field of the domain class to
     *                             deserialize.
     * @param codeAttributeEditor  the code attribute editor to be used for
     *                             injecting instructions.
     * @param builder              the instruction sequence builder to be used
     *                             for generating instructions.
     * @param gsonRuntimeSettings  tracks the GSON parameters that are utilized
     *                             in the code.
     */
    void deserialize(ProgramClass               programClass,
                     ProgramField               programField,
                     CodeAttributeEditor        codeAttributeEditor,
                     InstructionSequenceBuilder builder,
                     GsonRuntimeSettings        gsonRuntimeSettings);
}
