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
 * Class that groups all InlineDeserializer implementations for common types
 * together.
 *
 * @author Lars Vandenbergh
 */
class InlineDeserializers
{
    /**
     * Deserializer for handling primitive int, short and byte values.
     */
    static class InlinePrimitiveIntegerDeserializer implements InlineDeserializer
    {
        private final Class targetType;


        public InlinePrimitiveIntegerDeserializer()
        {
            this(null);
        }


        public InlinePrimitiveIntegerDeserializer(Class targetType)
        {
            this.targetType = targetType;
        }


        // Implementations for InlineDeserializer.

        @Override
        public boolean canDeserialize(GsonRuntimeSettings gsonRuntimeSettings)
        {
            return true;
        }

        @Override
        public void deserialize(ProgramClass               programClass,
                                ProgramField               programField,
                                CodeAttributeEditor        codeAttributeEditor,
                                InstructionSequenceBuilder ____,
                                GsonRuntimeSettings        gsonRuntimeSettings)
        {
            // Create labels for exception table.
            ConstantPoolEditor constantPoolEditor = new ConstantPoolEditor(programClass);
            int exceptionClassConstant =
                constantPoolEditor.addClassConstant(ClassConstants.NAME_JAVA_LANG_NUMBER_FORMAT_EXCEPTION,
                                                    null);

            CodeAttributeEditor.Label tryStart   = codeAttributeEditor.label();
            CodeAttributeEditor.Label tryEnd     = codeAttributeEditor.label();
            CodeAttributeEditor.Label catchStart =
                codeAttributeEditor.catch_(tryStart.offset(),
                                           tryEnd.offset(),
                                           exceptionClassConstant);
            CodeAttributeEditor.Label catchEnd   = codeAttributeEditor.label();

            // Try to read and parse integer.
            ____.label(tryStart)
                .aload(OptimizedClassConstants.FromJsonLocals.THIS)
                .aload(OptimizedClassConstants.FromJsonLocals.JSON_READER)
                .invokevirtual(GsonClassConstants.NAME_JSON_READER,
                               GsonClassConstants.METHOD_NAME_NEXT_INTEGER,
                               GsonClassConstants.METHOD_TYPE_NEXT_INTEGER);

            // Convert if necessary.
            if (targetType == byte.class)
            {
                ____.i2b();
            }
            else if (targetType == short.class)
            {
                ____.i2s();
            }

            //  Assign it to the field
            ____.putfield(programClass, programField)
                .goto_(catchEnd.offset())
                .label(tryEnd);

            // Throw JsonSyntaxException if reading and parsing the integer failed.
            int throwableLocal = OptimizedClassConstants.FromJsonLocals.MAX_LOCALS + 1;
            ____.label(catchStart)
                .astore(throwableLocal)
                .new_(GsonClassConstants.NAME_JSON_SYNTAX_EXCEPTION)
                .dup()
                .aload(throwableLocal)
                .invokespecial(GsonClassConstants.NAME_JSON_SYNTAX_EXCEPTION,
                               ClassConstants.METHOD_NAME_INIT,
                               ClassConstants.METHOD_TYPE_INIT_THROWABLE)
                .athrow()
                .label(catchEnd);
        }
    }


    /**
     * Deserializer for handling String values.
     */
    static class InlineStringDeserializer implements InlineDeserializer
    {
        // Implementations for InlineDeserializer.

        @Override
        public boolean canDeserialize(GsonRuntimeSettings gsonRuntimeSettings)
        {
            return true;
        }

        @Override
        public void deserialize(ProgramClass               programClass,
                                ProgramField               programField,
                                CodeAttributeEditor        codeAttributeEditor,
                                InstructionSequenceBuilder ____,
                                GsonRuntimeSettings        gsonRuntimeSettings)
        {
            CodeAttributeEditor.Label isBoolean = codeAttributeEditor.label();
            CodeAttributeEditor.Label end       = codeAttributeEditor.label();

            // Peek value and check whether it is a boolean.
            ____.aload(OptimizedClassConstants.FromJsonLocals.JSON_READER)
                .invokevirtual(GsonClassConstants.NAME_JSON_READER,
                               GsonClassConstants.METHOD_NAME_PEEK,
                               GsonClassConstants.METHOD_TYPE_PEEK)
                .getstatic(GsonClassConstants.NAME_JSON_TOKEN,
                           GsonClassConstants.FIELD_NAME_BOOLEAN,
                           GsonClassConstants.FIELD_TYPE_BOOLEAN)
                .ifacmpeq(isBoolean.offset());

            // It's not a boolean, just read the String and assign it to the
            // field.
            ____.aload(OptimizedClassConstants.FromJsonLocals.THIS)
                .aload(OptimizedClassConstants.FromJsonLocals.JSON_READER)
                .invokevirtual(GsonClassConstants.NAME_JSON_READER,
                               GsonClassConstants.METHOD_NAME_NEXT_STRING,
                               GsonClassConstants.METHOD_TYPE_NEXT_STRING)
                .putfield(programClass, programField)
                .goto_(end.offset());

            // It's a boolean, convert it to a String first and then assign it
            // to the field.
            ____.label(isBoolean)
                .aload(OptimizedClassConstants.FromJsonLocals.THIS)
                .aload(OptimizedClassConstants.FromJsonLocals.JSON_READER)
                .invokevirtual(GsonClassConstants.NAME_JSON_READER,
                               GsonClassConstants.METHOD_NAME_NEXT_BOOLEAN,
                               GsonClassConstants.METHOD_TYPE_NEXT_BOOLEAN)
                .invokestatic(ClassConstants.NAME_JAVA_LANG_BOOLEAN,
                              ClassConstants.METHOD_NAME_TOSTRING,
                              ClassConstants.METHOD_TYPE_TOSTRING_BOOLEAN)
                .putfield(programClass, programField)
                .label(end);
        }
    }
}
