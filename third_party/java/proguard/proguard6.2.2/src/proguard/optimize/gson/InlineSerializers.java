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
 * Class that groups all InlineSerializer implementations for common types
 * together.
 *
 * @author Lars Vandenbergh
 */
class InlineSerializers
{
    /**
     * Serializer for primitive boolean values.
     */
    static class InlinePrimitiveBooleanSerializer implements InlineSerializer
    {
        // Implementations for InlineSerializer.

        @Override
        public boolean canSerialize(ClassPool           programClassPool,
                                    GsonRuntimeSettings gsonRuntimeSettings)
        {
            return true;
        }


        @Override
        public void serialize(ProgramClass               programClass,
                              ProgramField               programField,
                              CodeAttributeEditor        codeAttributeEditor,
                              InstructionSequenceBuilder ____,
                              GsonRuntimeSettings        gsonRuntimeSettings)
        {
            ____.aload(OptimizedClassConstants.ToJsonLocals.JSON_WRITER)
                .aload(OptimizedClassConstants.ToJsonLocals.THIS)
                .getfield(programClass, programField)
                .invokevirtual(GsonClassConstants.NAME_JSON_WRITER,
                               GsonClassConstants.METHOD_NAME_VALUE_BOOLEAN,
                               GsonClassConstants.METHOD_TYPE_VALUE_BOOLEAN)
                .pop();
        }
    }

    /**
     * Serializer for handling Boolean values.
     */
    static class InlineBooleanSerializer implements InlineSerializer
    {
        // Implementations for InlineSerializer.

        @Override
        public boolean canSerialize(ClassPool           programClassPool,
                                    GsonRuntimeSettings gsonRuntimeSettings)
        {
            // Check whether JsonWriter.value(Boolean) is present in the used
            // version of Gson (should be from Gson 2.7 onwards).
            Clazz jsonWriterClass =
                programClassPool.getClass(GsonClassConstants.NAME_JSON_WRITER);
            Method valueBooleanMethod =
                jsonWriterClass.findMethod(GsonClassConstants.METHOD_NAME_VALUE_BOOLEAN_OBJECT,
                                           GsonClassConstants.METHOD_TYPE_VALUE_BOOLEAN_OBJECT);
            return valueBooleanMethod != null;
        }


        @Override
        public void serialize(ProgramClass               programClass,
                              ProgramField               programField,
                              CodeAttributeEditor        codeAttributeEditor,
                              InstructionSequenceBuilder ____,
                              GsonRuntimeSettings        gsonRuntimeSettings)
        {
            ____.aload(OptimizedClassConstants.ToJsonLocals.JSON_WRITER)
                .aload(OptimizedClassConstants.ToJsonLocals.THIS)
                .getfield(programClass, programField)
                .invokevirtual(GsonClassConstants.NAME_JSON_WRITER,
                               GsonClassConstants.METHOD_NAME_VALUE_BOOLEAN_OBJECT,
                               GsonClassConstants.METHOD_TYPE_VALUE_BOOLEAN_OBJECT)
                .pop();
        }
    }

    /**
     * Serializer for handling primitive int, short and byte values.
     */
    static class InlinePrimitiveIntegerSerializer implements InlineSerializer
    {
        // Implementations for InlineSerializer.

        @Override
        public boolean canSerialize(ClassPool           programClassPool,
                                    GsonRuntimeSettings gsonRuntimeSettings)
        {
            return true;
        }


        @Override
        public void serialize(ProgramClass               programClass,
                              ProgramField               programField,
                              CodeAttributeEditor        codeAttributeEditor,
                              InstructionSequenceBuilder ____,
                              GsonRuntimeSettings        gsonRuntimeSettings)
        {
            ____.aload(OptimizedClassConstants.ToJsonLocals.JSON_WRITER)
                .aload(OptimizedClassConstants.ToJsonLocals.THIS)
                .getfield(programClass, programField);
            ____.invokestatic(ClassConstants.NAME_JAVA_LANG_INTEGER,
                              ClassConstants.METHOD_NAME_VALUE_OF,
                              ClassConstants.METHOD_TYPE_VALUE_OF_INT);
            ____.invokevirtual(GsonClassConstants.NAME_JSON_WRITER,
                               GsonClassConstants.METHOD_NAME_VALUE_NUMBER,
                               GsonClassConstants.METHOD_TYPE_VALUE_NUMBER)
                .pop();
        }
    }

    /**
     * Serializer for handling String values.
     */
    static class InlineStringSerializer implements InlineSerializer
    {
        // Implementations for InlineSerializer.

        @Override
        public boolean canSerialize(ClassPool           programClassPool,
                                    GsonRuntimeSettings gsonRuntimeSettings)
        {
            return true;
        }


        @Override
        public void serialize(ProgramClass               programClass,
                              ProgramField               programField,
                              CodeAttributeEditor        codeAttributeEditor,
                              InstructionSequenceBuilder ____,
                              GsonRuntimeSettings        gsonRuntimeSettings)
        {
            ____.aload(OptimizedClassConstants.ToJsonLocals.JSON_WRITER)
                .aload(OptimizedClassConstants.ToJsonLocals.THIS)
                .getfield(programClass, programField)
                .invokevirtual(GsonClassConstants.NAME_JSON_WRITER,
                               GsonClassConstants.METHOD_NAME_VALUE_STRING,
                               GsonClassConstants.METHOD_TYPE_NAME_VALUE_STRING)
                .pop();
        }
    }
}
