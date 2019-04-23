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
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.editor.CodeAttributeEditor.Label;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;

import java.util.*;

/**
 * This class visitor transforms the template _OptimizedTypeAdapter into a full
 * implementation of a GSON TypeAdapter for a specific domain class.
 *
 * The read() and write() methods will be implemented appropriately and will
 * delegate to the toJson$xxx() and fromJson$xxx() methods that are injected
 * into the domain classes.
 *
 * The visited class will be renamed, e.g. if the domain class for which the
 * TypeAdapter is meant to be used is called DomainObject, the class name will
 * be OptimizedDomainObjectTypeAdapter.
 *
 * @author Lars Vandenbergh
 */
public class OptimizedTypeAdapterInitializer
extends      SimplifiedVisitor
implements   ClassVisitor
{
    private final String              typeAdapterClassName;
    private final String              objectClassName;
    private final ProgramClass        objectProgramClass;
    private final CodeAttributeEditor codeAttributeEditor;
    private final OptimizedJsonInfo   serializationIndexMap;
    private final OptimizedJsonInfo   deserializationIndexMap;
    private final ClassPool           instanceCreatorClasses;
    private final ClassVisitor        classVisitor;


    /**
     * Creates a new OptimizedTypeAdapterInitializer.
     *
     * @param typeAdapterClassName   the class name of the optimized type
     *                               adapter.
     * @param objectProgramClass     the class name of the domain class for
     *                               which the type adapter is meant.
     * @param codeAttributeEditor    the code attribute editor used to edit
     *                               the code attribute of the read() and
     *                               write() methods.
     * @param serializationInfo      contains information on which classes
     *                               and fields to serialize and how.
     * @param deserializationInfo    contains information on which classes
     *                               and fields to deserialize and how.
     * @param instanceCreatorClasses class pool that contains all domain
     *                               classes for which an InstanceCreator
     *                               is registered.
     * @param classVisitor           visitor to which all implemented type
     *                               adapters are delegated.
     */
    public OptimizedTypeAdapterInitializer(String              typeAdapterClassName,
                                           ProgramClass        objectProgramClass,
                                           CodeAttributeEditor codeAttributeEditor,
                                           OptimizedJsonInfo   serializationInfo,
                                           OptimizedJsonInfo   deserializationInfo,
                                           ClassPool           instanceCreatorClasses,
                                           ClassVisitor        classVisitor)
    {
        this.typeAdapterClassName    = typeAdapterClassName;
        this.objectClassName         = ClassUtil.internalClassName(objectProgramClass.getName());
        this.objectProgramClass      = objectProgramClass;
        this.codeAttributeEditor     = codeAttributeEditor;
        this.serializationIndexMap   = serializationInfo;
        this.deserializationIndexMap = deserializationInfo;
        this.instanceCreatorClasses  = instanceCreatorClasses;
        this.classVisitor            = classVisitor;
    }


    // Implementations for ClassVisitor.

    @Override
    public void visitAnyClass(Clazz clazz) {}


    @Override
    public void visitProgramClass(ProgramClass programClass)
    {
        // Rename template class to specific type adapter class name.
        programClass.thisClassConstantAccept(new TypeAdapterRenamer());
        programClass.methodsAccept(
            new AllAttributeVisitor(
            new AllAttributeVisitor(
            new LocalVariableTypeRenamer())));

        boolean isEnumAdapter = (objectProgramClass.getAccessFlags() & ClassConstants.ACC_ENUM) != 0;

        if (isEnumAdapter)
        {
            // Make sure the enum is accessible from the type adapter.
            objectProgramClass.u2accessFlags &= ~ClassConstants.ACC_PRIVATE;
            objectProgramClass.u2accessFlags |= ClassConstants.ACC_PUBLIC;
        }

        AttributeVisitor readImplementer  = isEnumAdapter ? new EnumReadImplementer():
                                                            new ReadImplementer();
        AttributeVisitor writeImplementer = isEnumAdapter ? new EnumWriteImplementer():
                                                            new WriteImplementer();

        if (deserializationIndexMap.classIndices.get(objectClassName) != null)
        {
            programClass.methodsAccept(new MemberNameFilter(OptimizedClassConstants.METHOD_NAME_READ,
                                       new AllAttributeVisitor(
                                       readImplementer)));
        }
        if (serializationIndexMap.classIndices.get(objectClassName) != null)
        {
            programClass.methodsAccept(new MemberNameFilter(OptimizedClassConstants.METHOD_NAME_WRITE,
                                       new AllAttributeVisitor(
                                       writeImplementer)));
        }

        // Pass on to class visitor.
        classVisitor.visitProgramClass(programClass);
    }


    private class TypeAdapterRenamer
    extends       SimplifiedVisitor
    implements    ConstantVisitor
    {

        // Implementations for ConstantVisitor.

        @Override
        public void visitAnyConstant(Clazz clazz, Constant constant) {}

        @Override
        public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
        {
            classConstant.u2nameIndex = new ConstantPoolEditor((ProgramClass)clazz).addUtf8Constant(typeAdapterClassName);
        }
    }


    private class LocalVariableTypeRenamer
    extends       SimplifiedVisitor
    implements    AttributeVisitor,
                  LocalVariableInfoVisitor
    {
        // Implementations for AttributeVisitor.

        @Override
        public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


        @Override
        public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
        {
            localVariableTableAttribute.localVariablesAccept(clazz,method,codeAttribute,this);
        }

        // Implementations for LocalVariableInfoVisitor.


        @Override
        public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
        {
            String descriptor = localVariableInfo.getDescriptor(clazz);

            if (descriptor.equals(OptimizedClassConstants.TYPE_OPTIMIZED_TYPE_ADAPTER_IMPL))
            {
                localVariableInfo.u2descriptorIndex = new ConstantPoolEditor((ProgramClass)clazz).addUtf8Constant(ClassUtil.internalTypeFromClassName(typeAdapterClassName));
            }
        }
    }


    /**
     * Visits the code attribute of the the read() method of the TypeAdapter and
     * provides it with a proper implementation for enum types.
     */
    private class EnumReadImplementer
    extends       SimplifiedVisitor
    implements    AttributeVisitor
    {
        // Implementations for AttributeVisitor.

        @Override
        public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


        @Override
        public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
        {
            codeAttributeEditor.reset(codeAttribute.u4codeLength);
            InstructionSequenceBuilder ____ = new InstructionSequenceBuilder((ProgramClass)clazz);

            // Deserialization
            Map<String, String[]> javaToJsonValueNames = deserializationIndexMap.classJsonInfos.get(objectClassName).javaToJsonFieldNames;
            Map<String, Integer>  jsonFieldIndices     = deserializationIndexMap.jsonFieldIndices;

            List<SwitchCase> switchCases = new ArrayList<SwitchCase>();

            for (String javaValueName : javaToJsonValueNames.keySet())
            {
                for (String jsonValueName : javaToJsonValueNames.get(javaValueName))
                {
                    switchCases.add(new SwitchCase(javaValueName,
                                                   codeAttributeEditor.label(),
                                                   jsonFieldIndices.get(jsonValueName)));
                }
            }

            Collections.sort(switchCases);

            int[] cases       = new int[switchCases.size()];
            int[] jumpOffsets = new int[switchCases.size()];

            for (int index = 0; index < switchCases.size(); index++)
            {
                cases[index]       = switchCases.get(index).stringIndex;
                jumpOffsets[index] = switchCases.get(index).label.offset();
            }

            CodeAttributeEditor.Label defaultCase = codeAttributeEditor.label();

            ____.aload(OptimizedClassConstants.ReadLocals.THIS)
                .getfield(typeAdapterClassName,
                          OptimizedClassConstants.FIELD_NAME_OPTIMIZED_JSON_READER,
                          OptimizedClassConstants.FIELD_TYPE_OPTIMIZED_JSON_READER)
                .aload(OptimizedClassConstants.ReadLocals.JSON_READER)
                .invokevirtual(OptimizedClassConstants.NAME_OPTIMIZED_JSON_READER,
                               OptimizedClassConstants.METHOD_NAME_NEXT_VALUE_INDEX,
                               OptimizedClassConstants.METHOD_TYPE_NEXT_VALUE_INDEX);

            ____.lookupswitch(defaultCase.offset(),
                              cases,
                              jumpOffsets);

            for (int index = 0; index < switchCases.size(); index++)
            {
                ____.label(switchCases.get(index).label)
                    .getstatic(objectClassName,
                               switchCases.get(index).enumConstant,
                               ClassUtil.internalTypeFromClassName(objectClassName))
                    .areturn();
            }

            ____.label(defaultCase)
                .aconst_null()
                .areturn();

            codeAttributeEditor.replaceInstruction(0, ____.instructions());
            codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
        }
    }


    private static class SwitchCase implements Comparable<SwitchCase>
    {
        private String enumConstant;
        private Label  label;
        private int    stringIndex;

        public SwitchCase(String enumConstant,
                          Label  label,
                          int    stringIndex)
        {
            this.enumConstant = enumConstant;
            this.label        = label;
            this.stringIndex  = stringIndex;
        }

        @Override
        public int compareTo(SwitchCase switchCase)
        {
            return this.stringIndex - switchCase.stringIndex;
        }
    }


    /**
     * Visits the code attribute of the the read() method of the TypeAdapter and
     * provides it with a proper implementation for non-enum types.
     */
    private class ReadImplementer
    extends       SimplifiedVisitor
    implements    AttributeVisitor
    {
        // Implementations for AttributeVisitor.

        @Override
        public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


        @Override
        public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
        {
            codeAttributeEditor.reset(codeAttribute.u4codeLength);
            InstructionSequenceBuilder ____ =
                new InstructionSequenceBuilder((ProgramClass)clazz);

            Integer classIndex         = deserializationIndexMap.classIndices.get(objectClassName);
            String  methodNameFromJson = OptimizedClassConstants.METHOD_NAME_FROM_JSON + classIndex;

            CodeAttributeEditor.Label nonNull = codeAttributeEditor.label();

            // Peek the next value and check if it is null.
            ____.aload(OptimizedClassConstants.ReadLocals.JSON_READER)
                .invokevirtual(GsonClassConstants.NAME_JSON_READER,
                               GsonClassConstants.METHOD_NAME_PEEK,
                               GsonClassConstants.METHOD_TYPE_PEEK);
            ____.getstatic(GsonClassConstants.NAME_JSON_TOKEN,
                           GsonClassConstants.FIELD_NAME_NULL,
                           GsonClassConstants.FIELD_TYPE_NULL)
                .ifacmpne(nonNull.offset());

            // If it is null, skip value in JSON.
            ____.aload(OptimizedClassConstants.ReadLocals.JSON_READER)
                .invokevirtual(GsonClassConstants.NAME_JSON_READER,
                               GsonClassConstants.METHOD_NAME_SKIP_VALUE,
                               GsonClassConstants.METHOD_TYPE_SKIP_VALUE);
            // Return null as result.
            ____.aconst_null()
                .areturn();

            // If the next value is not null, create an instance of the domain class.
            ____.label(nonNull);

            if (instanceCreatorClasses.getClass(objectClassName) == null)
            {
                ____.new_(objectClassName)
                    .dup()
                    .invokespecial(objectClassName,
                                   ClassConstants.METHOD_NAME_INIT,
                                   ClassConstants.METHOD_TYPE_INIT)
                    .astore(OptimizedClassConstants.ReadLocals.VALUE);
            }
            else
            {
                // For classes for which an InstanceCreator is registered, we
                // let the instance creator instantiate the class.
                ____.aload(OptimizedClassConstants.ReadLocals.THIS)
                    .getfield(typeAdapterClassName,
                              OptimizedClassConstants.FIELD_NAME_GSON,
                              OptimizedClassConstants.FIELD_TYPE_GSON)
                    .getfield(GsonClassConstants.NAME_GSON,
                              GsonClassConstants.FIELD_NAME_INSTANCE_CREATORS,
                              GsonClassConstants.FIELD_TYPE_INSTANCE_CREATORS)
                    .ldc(instanceCreatorClasses.getClass(objectClassName))
                    .invokevirtual(ClassConstants.NAME_JAVA_UTIL_MAP,
                                   ClassConstants.METHOD_NAME_MAP_GET,
                                   ClassConstants.METHOD_TYPE_MAP_GET)
                    .checkcast(GsonClassConstants.NAME_INSTANCE_CREATOR)
                    .ldc(instanceCreatorClasses.getClass(objectClassName))
                    .invokevirtual(GsonClassConstants.NAME_INSTANCE_CREATOR,
                                   GsonClassConstants.METHOD_NAME_CREATE_INSTANCE,
                                   GsonClassConstants.METHOD_TYPE_CREATE_INSTANCE)
                    .checkcast(objectClassName)
                    .astore(OptimizedClassConstants.ReadLocals.VALUE);
            }

            // Deserialize object by calling its fromJson$ method.
            ____.aload(OptimizedClassConstants.ReadLocals.VALUE)
                .aload(OptimizedClassConstants.ReadLocals.THIS)
                .getfield(typeAdapterClassName,
                          OptimizedClassConstants.FIELD_NAME_GSON,
                          OptimizedClassConstants.FIELD_TYPE_GSON)
                .aload(OptimizedClassConstants.ReadLocals.JSON_READER)
                .aload(OptimizedClassConstants.ReadLocals.THIS)
                .getfield(typeAdapterClassName,
                          OptimizedClassConstants.FIELD_NAME_OPTIMIZED_JSON_READER,
                          OptimizedClassConstants.FIELD_TYPE_OPTIMIZED_JSON_READER)
                .invokevirtual(objectClassName,
                               methodNameFromJson,
                               OptimizedClassConstants.METHOD_TYPE_FROM_JSON);

            // Return deserialized object.
            ____.aload(OptimizedClassConstants.ReadLocals.VALUE)
                .areturn();

            codeAttributeEditor.replaceInstruction(0, ____.instructions());
            codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
        }
    }


    /**
     * Visits the code attribute of the the write() method of the TypeAdapter and
     * provides it with a proper implementation for enum types.
     */
    private class EnumWriteImplementer
    extends       SimplifiedVisitor
    implements    AttributeVisitor
    {
        // Implementations for AttributeVisitor.

        @Override
        public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


        @Override
        public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
        {
            codeAttributeEditor.reset(codeAttribute.u4codeLength);
            InstructionSequenceBuilder ____ = new InstructionSequenceBuilder((ProgramClass)clazz);

            Label nonNull    = codeAttributeEditor.label();
            Label end        = codeAttributeEditor.label();
            Label writeValue = codeAttributeEditor.label();

            // Check if the passed value is null.
            ____.aload(OptimizedClassConstants.WriteLocals.VALUE)
                .ifnonnull(nonNull.offset());

            // If it is null, write null value in JSON.
            ____.aload(OptimizedClassConstants.WriteLocals.JSON_WRITER)
                .invokevirtual(GsonClassConstants.NAME_JSON_WRITER,
                               GsonClassConstants.METHOD_NAME_NULL_VALUE,
                               GsonClassConstants.METHOD_TYPE_NULL_VALUE)
                .pop()
                .goto_(end.offset());

            // If it is not null, serialize value.
            Map<String, String[]> javaToJsonValueNames = serializationIndexMap.classJsonInfos.get(objectClassName).javaToJsonFieldNames;
            Map<String, Integer>  jsonFieldIndices     = serializationIndexMap.jsonFieldIndices;

            ____.label(nonNull)
                .aload(OptimizedClassConstants.WriteLocals.THIS)
                .getfield(typeAdapterClassName,
                          OptimizedClassConstants.FIELD_NAME_OPTIMIZED_JSON_WRITER,
                          OptimizedClassConstants.FIELD_TYPE_OPTIMIZED_JSON_WRITER)
                .aload(OptimizedClassConstants.WriteLocals.JSON_WRITER)
                .aload(OptimizedClassConstants.WriteLocals.VALUE);

            for (String javaValueName: javaToJsonValueNames.keySet())
            {
                Label  label         = codeAttributeEditor.label();
                String jsonValueName = javaToJsonValueNames.get(javaValueName)[0];

                ____.dup()
                    .getstatic(objectClassName,
                               javaValueName,
                               ClassUtil.internalTypeFromClassName(objectClassName))
                    .ifacmpne(label.offset())
                    .pop()
                    .ldc(jsonFieldIndices.get(jsonValueName).intValue())
                    .goto_(writeValue.offset())
                    .label(label);
            }
            ____.pop()
                .iconst_m1();

            ____.label(writeValue)
                .invokevirtual(OptimizedClassConstants.NAME_OPTIMIZED_JSON_WRITER,
                               OptimizedClassConstants.METHOD_NAME_VALUE,
                               OptimizedClassConstants.METHOD_TYPE_VALUE)
                .label(end)
                .return_();

            codeAttributeEditor.replaceInstruction(0, ____.instructions());
            codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
        }
    }


    /**
     * Visits the code attribute of the the write() method of the TypeAdapter and
     * provides it with a proper implementation for non-enum types.
     */
    private class WriteImplementer
    extends       SimplifiedVisitor
    implements    AttributeVisitor
    {
        // Implementations for AttributeVisitor.

        @Override
        public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


        @Override
        public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
        {
            codeAttributeEditor.reset(codeAttribute.u4codeLength);
            InstructionSequenceBuilder ____ =
                new InstructionSequenceBuilder((ProgramClass)clazz);

            Integer classIndex       = serializationIndexMap.classIndices.get(objectClassName);
            String  methodNameToJson = OptimizedClassConstants.METHOD_NAME_TO_JSON + classIndex;

            CodeAttributeEditor.Label nonNull = codeAttributeEditor.label();
            CodeAttributeEditor.Label end     = codeAttributeEditor.label();

            // Check if the passed value is null.
            ____.aload(OptimizedClassConstants.WriteLocals.VALUE)
                .ifnonnull(nonNull.offset());

            // If it is null, write null value in JSON.
            ____.aload(OptimizedClassConstants.WriteLocals.JSON_WRITER)
                .invokevirtual(GsonClassConstants.NAME_JSON_WRITER,
                               GsonClassConstants.METHOD_NAME_NULL_VALUE,
                               GsonClassConstants.METHOD_TYPE_NULL_VALUE)
                .pop()
                .goto_(end.offset());

            // If the next value is not null, serialize object by calling its toJson$ method.
            ____.label(nonNull)
                .aload(OptimizedClassConstants.WriteLocals.VALUE)
                .checkcast(objectClassName)
                .aload(OptimizedClassConstants.WriteLocals.THIS)
                .getfield(typeAdapterClassName,
                          OptimizedClassConstants.FIELD_NAME_GSON,
                          OptimizedClassConstants.FIELD_TYPE_GSON)
                .aload(OptimizedClassConstants.WriteLocals.JSON_WRITER)
                .aload(OptimizedClassConstants.WriteLocals.THIS)
                .getfield(typeAdapterClassName,
                          OptimizedClassConstants.FIELD_NAME_OPTIMIZED_JSON_WRITER,
                          OptimizedClassConstants.FIELD_TYPE_OPTIMIZED_JSON_WRITER)
                .invokevirtual(objectClassName,
                               methodNameToJson,
                               OptimizedClassConstants.METHOD_TYPE_TO_JSON)
                .label(end)
                .return_();

            codeAttributeEditor.replaceInstruction(0, ____.instructions());
            codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
        }
    }
}
