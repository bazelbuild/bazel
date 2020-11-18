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
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;

import static proguard.classfile.instruction.InstructionConstants.OP_ARETURN;
import static proguard.classfile.instruction.InstructionConstants.OP_RETURN;

/**
 * Instruction visitor that adds some instrumentation code to the Gson.toJson()
 * and Gson.fromJson() methods that prints out the type adapter cache. This
 * can be useful for debugging purposes.
 *
 * @author Lars Vandenbergh
 */
public class GsonInstrumentationAdder
extends      SimplifiedVisitor
implements   InstructionVisitor
{
    private static final boolean DEBUG = false;

    private final ClassPool           programClassPool;
    private final ClassPool           libraryClassPool;
    private final CodeAttributeEditor codeAttributeEditor;


    /**
     * Creates a new GsonInstrumentationAdder.
     *
     * @param programClassPool     the program class pool used for looking up
     *                             program class references.
     * @param libraryClassPool     the library class pool used for looking up
     *                             library class references.
     * @param codeAttributeEditor  the code attribute editor used for editing
     *                             the code attribute of the Gson methods.
     */
    public GsonInstrumentationAdder(ClassPool           programClassPool,
                                    ClassPool           libraryClassPool,
                                    CodeAttributeEditor codeAttributeEditor)
    {
        this.programClassPool    = programClassPool;
        this.libraryClassPool    = libraryClassPool;
        this.codeAttributeEditor = codeAttributeEditor;
    }


    // Implementations for InstructionVisitor.

    @Override
    public void visitAnyInstruction(Clazz         clazz,
                                    Method        method,
                                    CodeAttribute codeAttribute,
                                    int           offset,
                                    Instruction   instruction)
    {
        if (instruction.actualOpcode() == OP_RETURN ||
            instruction.actualOpcode() == OP_ARETURN)
        {
            String fullyQualifiedMethodName = clazz.getName() + "#" +
                                              method.getName(clazz) + method.getDescriptor(clazz);
            if (DEBUG)
            {
                System.out.println("GsonInstrumentationAdder: instrumenting " +
                                   fullyQualifiedMethodName);
            }

            InstructionSequenceBuilder ____ = new InstructionSequenceBuilder((ProgramClass)clazz,
                                                                             programClassPool,
                                                                             libraryClassPool);
            ____.ldc("Type token cache after invoking " + fullyQualifiedMethodName + ":")
                .aload_0()
                .getfield(clazz.getName(),
                          GsonClassConstants.FIELD_NAME_TYPE_TOKEN_CACHE,
                          GsonClassConstants.FIELD_TYPE_TYPE_TOKEN_CACHE)
                .invokestatic(OptimizedClassConstants.NAME_GSON_UTIL,
                              OptimizedClassConstants.METHOD_NAME_DUMP_TYPE_TOKEN_CACHE,
                              OptimizedClassConstants.METHOD_TYPE_DUMP_TYPE_TOKEN_CACHE);

            codeAttributeEditor.insertBeforeInstruction(offset, ____.instructions());
        }
    }
}
