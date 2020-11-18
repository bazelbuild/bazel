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
 * _OptimizedJsonReaderImpl so that the data structure is initialized
 * with the correct mapping between Json field names and internal
 * indices.
 *
 * @author Lars Vandenbergh
 */
public class OptimizedJsonReaderImplInitializer
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    private final ClassPool           programClassPool;
    private final ClassPool           libraryClassPool;
    private final CodeAttributeEditor codeAttributeEditor;
    private final OptimizedJsonInfo   deserializationInfo;


    /**
     * Creates a new OptimizedJsonReaderImplInitializer.
     *
     * @param programClassPool    the program class pool used for looking up
     *                            program class references.
     * @param libraryClassPool    the library class pool used for looking up
     *                            library class references.
     * @param codeAttributeEditor the code attribute editor used for editing
     *                            the code attribute of the static initializer.
     * @param deserializationInfo contains information on which classes and
     *                            fields to deserialize and how.
     */
    public OptimizedJsonReaderImplInitializer(ClassPool           programClassPool,
                                              ClassPool           libraryClassPool,
                                              CodeAttributeEditor codeAttributeEditor,
                                              OptimizedJsonInfo   deserializationInfo)
    {
        this.programClassPool    = programClassPool;
        this.libraryClassPool    = libraryClassPool;
        this.codeAttributeEditor = codeAttributeEditor;
        this.deserializationInfo = deserializationInfo;
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

        ____.new_(ClassConstants.NAME_JAVA_UTIL_HASH_MAP, libraryClassPool.getClass(ClassConstants.NAME_JAVA_UTIL_HASH_MAP))
            .dup()
            .invokespecial(ClassConstants.NAME_JAVA_UTIL_HASH_MAP,
                           ClassConstants.METHOD_NAME_INIT,
                           ClassConstants.METHOD_TYPE_INIT);

        for (Map.Entry<String, Integer> jsonFieldIndicesEntry : deserializationInfo.jsonFieldIndices.entrySet())
        {
            ____.dup()
                .ldc(jsonFieldIndicesEntry.getKey())
                .ldc(jsonFieldIndicesEntry.getValue().intValue())
                .invokestatic(ClassConstants.NAME_JAVA_LANG_INTEGER,
                              ClassConstants.METHOD_NAME_VALUE_OF,
                              ClassConstants.METHOD_TYPE_VALUE_OF_INT)
                .invokevirtual(ClassConstants.NAME_JAVA_UTIL_HASH_MAP,
                               ClassConstants.METHOD_NAME_MAP_PUT,
                               ClassConstants.METHOD_TYPE_MAP_PUT)
                .pop();
        }

        // We replace the instruction that loads null on the stack with the
        // initialization code and leave the return instruction that comes
        // right after it in place.
        codeAttributeEditor.replaceInstruction(0, ____.instructions());
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
    }
}
