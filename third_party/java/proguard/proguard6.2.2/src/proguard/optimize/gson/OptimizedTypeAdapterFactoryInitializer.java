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

import java.lang.reflect.Modifier;
import java.util.Map;

/**
 * This visitor implements the getType() method of the injected
 * _OptimizedTypeAdapterFactory.
 *
 * @author Lars Vandenbergh
 */
public class OptimizedTypeAdapterFactoryInitializer
extends      SimplifiedVisitor
implements   InstructionVisitor
{
    private final ClassPool           programClassPool;
    private final CodeAttributeEditor codeAttributeEditor;
    private final Map<String, String> typeAdapterRegistry;
    private final GsonRuntimeSettings gsonRuntimeSettings;


    /**
     * Creates a new OptimizedTypeAdapterFactoryInitializer.
     *
     * @param programClassPool    the program class pool used for looking up
     *                            references to program classes.
     * @param codeAttributeEditor the code attribute editor used for editing
     *                            the code attribute of the getType() method.
     * @param typeAdapterRegistry contains the mapping between domain class
     *                            names and their corresponding type adapter
     *                            class name.
     * @param gsonRuntimeSettings keeps track of all GsonBuilder invocations.
     */
    public OptimizedTypeAdapterFactoryInitializer(ClassPool           programClassPool,
                                                  CodeAttributeEditor codeAttributeEditor,
                                                  Map<String, String> typeAdapterRegistry,
                                                  GsonRuntimeSettings gsonRuntimeSettings)
    {
        this.programClassPool    = programClassPool;
        this.codeAttributeEditor = codeAttributeEditor;
        this.typeAdapterRegistry = typeAdapterRegistry;
        this.gsonRuntimeSettings = gsonRuntimeSettings;
    }


    // Implementations for InstructionVisitor.

    @Override
    public void visitAnyInstruction(Clazz         clazz,
                                    Method        method,
                                    CodeAttribute codeAttribute,
                                    int           offset,
                                    Instruction   instruction)
    {
        if (method.getName(clazz).equals(OptimizedClassConstants.METHOD_NAME_CREATE)       &&
            method.getDescriptor(clazz).equals(OptimizedClassConstants.METHOD_TYPE_CREATE) &&
            instruction.actualOpcode() == InstructionConstants.OP_ACONST_NULL)
        {
            InstructionSequenceBuilder ____ =
                new InstructionSequenceBuilder((ProgramClass)clazz);
            CodeAttributeEditor.Label end = codeAttributeEditor.label();

            // Don't use optimized type adapter when an unsupported property is
            // enabled during the construction of the Gson object.
            if (gsonRuntimeSettings.setFieldNamingPolicy ||
                gsonRuntimeSettings.setFieldNamingStrategy)
            {
                ____.aload_1()  // gson argument
                    .getfield(GsonClassConstants.NAME_GSON,
                              GsonClassConstants.FIELD_NAME_FIELD_NAMING_STRATEGY,
                              GsonClassConstants.FIELD_TYPE_FIELD_NAMING_STRATEGY)
                    .getstatic(GsonClassConstants.NAME_FIELD_NAMING_POLICY,
                               GsonClassConstants.FIELD_NAME_IDENTITY,
                               GsonClassConstants.FIELD_TYPE_IDENTITY)
                    .ifacmpne(end.offset());
            }
            if (gsonRuntimeSettings.excludeFieldsWithModifiers)
            {
                ____.aload_1()  // gson argument
                    .getfield(GsonClassConstants.NAME_GSON,
                              GsonClassConstants.FIELD_NAME_EXCLUDER,
                              GsonClassConstants.FIELD_TYPE_EXCLUDER)
                    .getfield(GsonClassConstants.NAME_EXCLUDER,
                              GsonClassConstants.FIELD_NAME_MODIFIERS,
                              GsonClassConstants.FIELD_TYPE_MODIFIERS)
                    .ldc(Modifier.STATIC | Modifier.TRANSIENT)
                    .ificmpne(end.offset());
            }
            if (gsonRuntimeSettings.setExclusionStrategies ||
                gsonRuntimeSettings.addSerializationExclusionStrategy ||
                gsonRuntimeSettings.addDeserializationExclusionStrategy)
            {
                ____.aload_1()  // gson argument
                    .getfield(GsonClassConstants.NAME_GSON,
                              GsonClassConstants.FIELD_NAME_EXCLUDER,
                              GsonClassConstants.FIELD_TYPE_EXCLUDER)
                    .getfield(GsonClassConstants.NAME_EXCLUDER,
                              GsonClassConstants.FIELD_NAME_SERIALIZATION_STRATEGIES,
                              GsonClassConstants.FIELD_TYPE_SERIALIZATION_STRATEGIES)
                    .invokevirtual(ClassConstants.NAME_JAVA_UTIL_LIST,
                                   ClassConstants.METHOD_NAME_IS_EMPTY,
                                   ClassConstants.METHOD_TYPE_IS_EMPTY)
                    .ifeq(end.offset());
                ____.aload_1()  // gson argument
                    .getfield(GsonClassConstants.NAME_GSON,
                              GsonClassConstants.FIELD_NAME_EXCLUDER,
                              GsonClassConstants.FIELD_TYPE_EXCLUDER)
                    .getfield(GsonClassConstants.NAME_EXCLUDER,
                              GsonClassConstants.FIELD_NAME_DESERIALIZATION_STRATEGIES,
                              GsonClassConstants.FIELD_TYPE_DESERIALIZATION_STRATEGIES)
                    .invokevirtual(ClassConstants.NAME_JAVA_UTIL_LIST,
                                   ClassConstants.METHOD_NAME_IS_EMPTY,
                                   ClassConstants.METHOD_TYPE_IS_EMPTY)
                    .ifeq(end.offset());
            }
            if (gsonRuntimeSettings.setVersion)
            {
                ____.aload_1()  // gson argument
                    .getfield(GsonClassConstants.NAME_GSON,
                              GsonClassConstants.FIELD_NAME_EXCLUDER,
                              GsonClassConstants.FIELD_TYPE_EXCLUDER)
                    .getfield(GsonClassConstants.NAME_EXCLUDER,
                              GsonClassConstants.FIELD_NAME_VERSION,
                              GsonClassConstants.FIELD_TYPE_VERSION)
                    .ldc2_w(-1d)
                    .dcmpg()
                    .ifne(end.offset());
            }

            for (Map.Entry<String, String> typeAdapterRegistryEntry : typeAdapterRegistry.entrySet())
            {
                String objectType  = typeAdapterRegistryEntry.getKey();
                String adapterType = typeAdapterRegistryEntry.getValue();
                Clazz objectClazz  = programClassPool.getClass(objectType);

                CodeAttributeEditor.Label elseif = codeAttributeEditor.label();
                ____.aload_2()  // type argument
                    .invokevirtual(GsonClassConstants.NAME_TYPE_TOKEN,
                                   GsonClassConstants.METHOD_NAME_GET_RAW_TYPE,
                                   GsonClassConstants.METHOD_TYPE_GET_RAW_TYPE)
                    .ldc(objectClazz)
                    .ifacmpne(elseif.offset())
                    .new_(adapterType)
                    .dup()
                    .aload_1()  // gson argument
                    .getstatic(clazz.getName(),
                               OptimizedClassConstants.FIELD_NAME_OPTIMIZED_JSON_READER_IMPL,
                               OptimizedClassConstants.FIELD_TYPE_OPTIMIZED_JSON_READER_IMPL)
                    .getstatic(clazz.getName(),
                               OptimizedClassConstants.FIELD_NAME_OPTIMIZED_JSON_WRITER_IMPL,
                               OptimizedClassConstants.FIELD_TYPE_OPTIMIZED_JSON_WRITER_IMPL)
                    .invokespecial(adapterType,
                                   ClassConstants.METHOD_NAME_INIT,
                                   OptimizedClassConstants.METHOD_TYPE_INIT)
                    .areturn()
                    .label(elseif);
            }

            ____.label(end)
                .aconst_null();

            codeAttributeEditor.replaceInstruction(offset,____.instructions());
        }
    }
}
