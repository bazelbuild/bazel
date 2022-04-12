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
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.util.SimplifiedVisitor;

import java.util.Map;

/**
 * This code attribute visitor implements the static initializer of
 * _OptimizedJsonWriterImpl so that the data structure is initialized
 * with the correct mapping between internal indices and Json field names.
 *
 * @author Lars Vandenbergh
 */
public class OptimizedJsonWriterImplInitializer
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    private final ClassPool           programClassPool;
    private final ClassPool           libraryClassPool;
    private final CodeAttributeEditor codeAttributeEditor;
    private final OptimizedJsonInfo   serializationInfo;


    /**
     * Creates a new OptimizedJsonWriterImplInitializer.
     *
     * @param programClassPool    the program class pool used for looking up
     *                            program class references.
     * @param libraryClassPool    the library class pool used for looking up
     *                            library class references.
     * @param codeAttributeEditor the code attribute editor used for editing
     *                            the code attribute of the static initializer.
     * @param serializationInfo   contains information on which classes and
     *                            fields to serialize and how.
     */
    public OptimizedJsonWriterImplInitializer(ClassPool           programClassPool,
                                              ClassPool           libraryClassPool,
                                              CodeAttributeEditor codeAttributeEditor,
                                              OptimizedJsonInfo   serializationInfo)
    {
        this.programClassPool    = programClassPool;
        this.libraryClassPool    = libraryClassPool;
        this.codeAttributeEditor = codeAttributeEditor;
        this.serializationInfo   = serializationInfo;
    }


    // Implementations for AttributeVisitor.

    @Override
    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    @Override
    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        codeAttributeEditor.reset(codeAttribute.u4codeLength);

        InstructionSequenceBuilder ____ =
            new InstructionSequenceBuilder((ProgramClass)clazz,
                                           programClassPool,
                                           libraryClassPool);

        Map<String, Integer> fieldIndices = serializationInfo.jsonFieldIndices;

        ____.ldc(fieldIndices.size())
            .anewarray(ClassConstants.NAME_JAVA_LANG_STRING,
                       libraryClassPool.getClass(ClassConstants.NAME_JAVA_LANG_STRING));

        for (Map.Entry<String, Integer> fieldIndexEntry : fieldIndices.entrySet())
        {
            ____.dup()
                .ldc(fieldIndexEntry.getValue().intValue())
                .ldc(fieldIndexEntry.getKey())
                .aastore();
        }

        // We replace the instruction that loads null on the stack with the
        // initialization code and leave the return instruction that comes
        // right after it in place.
        codeAttributeEditor.replaceInstruction(0, ____.instructions());
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
    }
}
