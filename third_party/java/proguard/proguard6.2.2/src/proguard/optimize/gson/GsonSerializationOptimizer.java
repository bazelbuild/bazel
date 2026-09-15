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
import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.optimize.info.ProgramMemberOptimizationInfoSetter;
import proguard.util.MultiValueMap;

import java.util.*;

import static proguard.optimize.gson.OptimizedClassConstants.*;

/**
 * This visitor injects a toJson$xxx() method into the classes that it visits
 * that serializes its fields to Json.
 *
 * @author Lars Vandenbergh
 * @author Rob Coekaerts
 */
public class GsonSerializationOptimizer
extends      SimplifiedVisitor
implements   MemberVisitor,
             ClassVisitor,
             ElementValueVisitor,
             AttributeVisitor,
             AnnotationVisitor
{
    private static final boolean DEBUG = false;

    private static final Map<String,InlineSerializer> inlineSerializers = new HashMap<String, InlineSerializer>();

    private final ClassPool                     programClassPool;
    private final ClassPool                     libraryClassPool;
    private final GsonRuntimeSettings           gsonRuntimeSettings;
    private final CodeAttributeEditor           codeAttributeEditor;
    private final OptimizedJsonInfo             serializationInfo;
    private final boolean                       supportExposeAnnotation;
    private final MultiValueMap<String, String> injectedClassNameMap;

    private InstructionSequenceBuilder ____;

    static
    {
        inlineSerializers.put(ClassConstants.TYPE_BOOLEAN + "",
                              new InlineSerializers.InlinePrimitiveBooleanSerializer());
        inlineSerializers.put(ClassConstants.TYPE_JAVA_LANG_BOOLEAN,
                              new InlineSerializers.InlineBooleanSerializer());
        inlineSerializers.put(ClassConstants.TYPE_BYTE + "",
                              new InlineSerializers.InlinePrimitiveIntegerSerializer());
        inlineSerializers.put(ClassConstants.TYPE_SHORT + "",
                              new InlineSerializers.InlinePrimitiveIntegerSerializer());
        inlineSerializers.put(ClassConstants.TYPE_INT + "",
                              new InlineSerializers.InlinePrimitiveIntegerSerializer());
        inlineSerializers.put(ClassConstants.TYPE_JAVA_LANG_STRING,
                              new InlineSerializers.InlineStringSerializer());
    }

    /**
     * Creates a new GsonSerializationOptimizer.
     *
     * @param programClassPool      the program class pool to initialize
     *                              added references.
     * @param libraryClassPool      the library class pool to initialize
     *                              added references.
     * @param gsonRuntimeSettings   keeps track of all GsonBuilder
 *                                  invocations.
     * @param codeAttributeEditor   the code editor that is used to
     *                              inject optimized code into the domain
     *                              classes.
     * @param serializationInfo     contains information on which class
     *                              and fields need to be optimized and how.
     * @param injectedClassNameMap the map that keeps track of injected
     *                              classes.
     */
    public GsonSerializationOptimizer(ClassPool                     programClassPool,
                                      ClassPool                     libraryClassPool,
                                      GsonRuntimeSettings           gsonRuntimeSettings,
                                      CodeAttributeEditor           codeAttributeEditor,
                                      OptimizedJsonInfo             serializationInfo,
                                      MultiValueMap<String, String> injectedClassNameMap)
    {
        this.programClassPool        = programClassPool;
        this.libraryClassPool        = libraryClassPool;
        this.gsonRuntimeSettings     = gsonRuntimeSettings;
        this.codeAttributeEditor     = codeAttributeEditor;
        this.serializationInfo       = serializationInfo;
        this.supportExposeAnnotation = gsonRuntimeSettings.excludeFieldsWithoutExposeAnnotation;
        this.injectedClassNameMap    = injectedClassNameMap;
    }


    // Implementations for ClassVisitor.

    public void visitAnyClass(Clazz clazz) {}


    public void visitProgramClass(ProgramClass programClass)
    {
        // Make access public for _OptimizedTypeAdapterFactory.
        programClass.u2accessFlags &= ~ClassConstants.ACC_PRIVATE;
        programClass.u2accessFlags |= ClassConstants.ACC_PUBLIC;

        // Start adding new serialization methods.
        SimplifiedClassEditor classEditor =
            new SimplifiedClassEditor(programClass);

        ____ = new InstructionSequenceBuilder(programClass,
                                              programClassPool,
                                              libraryClassPool);

        // Add toJson$ method.
        Integer classIndex           = serializationInfo.classIndices.get(programClass.getName());
        String  methodNameToJson     = METHOD_NAME_TO_JSON             + classIndex;
        String  methodNameToJsonBody = METHOD_NAME_TO_JSON_BODY        + classIndex;

        if (DEBUG)
        {
            System.out.println(
                "GsonSerializationOptimizer: adding " +
                methodNameToJson +
                " method to " + programClass.getName());
        }

        ProgramMethod toJsonMethod = classEditor.addMethod(
            ClassConstants.ACC_PUBLIC | ClassConstants.ACC_SYNTHETIC,
            methodNameToJson,
            METHOD_TYPE_TO_JSON,
            ____.return_()
                .instructions());

        toJsonMethod.accept(programClass,
                            new ProgramMemberOptimizationInfoSetter());

        // Create new CodeAttributeEditor for the toJson$ method.
        codeAttributeEditor.reset(1);

        // Begin Json object.
        ____.aload(OptimizedClassConstants.ToJsonLocals.JSON_WRITER)
            .invokevirtual(GsonClassConstants.NAME_JSON_WRITER,
                           GsonClassConstants.METHOD_NAME_WRITER_BEGIN_OBJECT,
                           GsonClassConstants.METHOD_TYPE_WRITER_BEGIN_OBJECT);

        // Invoke toJsonBody$.
        ____.aload(OptimizedClassConstants.ToJsonLocals.THIS)
            .aload(OptimizedClassConstants.ToJsonLocals.GSON)
            .aload(OptimizedClassConstants.ToJsonLocals.JSON_WRITER)
            .aload(OptimizedClassConstants.ToJsonLocals.OPTIMIZED_JSON_WRITER)
            .invokevirtual(programClass.getName(),
                           methodNameToJsonBody,
                           METHOD_TYPE_TO_JSON_BODY);

        // End Json object.
        ____.aload(OptimizedClassConstants.ToJsonLocals.JSON_WRITER)
            .invokevirtual(GsonClassConstants.NAME_JSON_WRITER,
                           GsonClassConstants.METHOD_NAME_WRITER_END_OBJECT,
                           GsonClassConstants.METHOD_TYPE_WRITER_END_OBJECT)
            .return_();

        // Add all toJson$ instructions.
        codeAttributeEditor.replaceInstruction(0, ____.instructions());
        toJsonMethod.attributesAccept(programClass, codeAttributeEditor);

        addToJsonBodyMethod(programClass, classEditor);

        programClass.accept(new MethodLinker());

        classEditor.finishEditing(programClassPool,
                                  libraryClassPool);
    }


    private void addToJsonBodyMethod(ProgramClass          programClass,
                                     SimplifiedClassEditor classEditor)
    {
        Integer classIndex = serializationInfo.classIndices.get(programClass.getName());
        String  methodName = METHOD_NAME_TO_JSON_BODY + classIndex;

        // Add toJsonBody$ method.
        if (DEBUG)
        {
            System.out.println(
                "GsonSerializationOptimizer: adding " +
                methodName +
                " method to " + programClass.getName());
        }


        ProgramMethod toJsonBodyMethod = classEditor.addMethod(
            ClassConstants.ACC_PROTECTED | ClassConstants.ACC_SYNTHETIC,
            methodName,
            METHOD_TYPE_TO_JSON_BODY,
            ____.return_()
                .instructions());

        // Add optimization info to new method.
        toJsonBodyMethod.accept(programClass,
                                new ProgramMemberOptimizationInfoSetter());

        // Edit code attribute of fromJson$.
        toJsonBodyMethod.attributesAccept(programClass,
                                        new ToJsonCodeAttributeVisitor());
    }


    private class ToJsonCodeAttributeVisitor
    extends       SimplifiedVisitor
    implements    AttributeVisitor,
                  MemberVisitor
    {
        private int valueLocalIndex;

        // Implementations for AttributeVisitor.

        @Override
        public void visitCodeAttribute(Clazz         clazz,
                                       Method        method,
                                       CodeAttribute codeAttribute)
        {
            // Create new CodeAttributeEditor for the toJsonBody$ method.
            codeAttributeEditor.reset(1);

            // Assign locals for nextFieldIndex and isNull.
            valueLocalIndex = codeAttribute.u2maxLocals;

            // Apply non static member visitor to all fields to visit.
            clazz.fieldsAccept(new MemberAccessFilter(0,
                                                             ClassConstants.ACC_SYNTHETIC |
                                                             ClassConstants.ACC_STATIC,
                                                             this));

            // Call the superclass toJsonBody$ if there is one.
            if (!clazz.getSuperClass().getName().equals(ClassConstants.NAME_JAVA_LANG_OBJECT))
            {
                Integer superClassIndex =
                    serializationInfo.classIndices.get(clazz.getSuperClass().getName());
                String superMethodNameToJsonBody = METHOD_NAME_TO_JSON_BODY + superClassIndex;

                ____.aload(OptimizedClassConstants.ToJsonLocals.THIS)
                    .aload(OptimizedClassConstants.ToJsonLocals.GSON)
                    .aload(OptimizedClassConstants.ToJsonLocals.JSON_WRITER)
                    .aload(OptimizedClassConstants.ToJsonLocals.OPTIMIZED_JSON_WRITER)
                    .invokevirtual(clazz.getSuperClass().getName(),
                                   superMethodNameToJsonBody,
                                   METHOD_TYPE_TO_JSON_BODY);
            }

            ____.return_();

            // Add all toJsonBody$ instructions.
            codeAttributeEditor.replaceInstruction(0, ____.instructions());
            codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
        }


        // Implementations for MemberVisitor.

        public void visitProgramField(ProgramClass programClass,
                                      ProgramField programField)
        {
            OptimizedJsonInfo.ClassJsonInfo classSerializationInfo =
                serializationInfo.classJsonInfos.get(programClass.getName());
            String[] jsonFieldNames = classSerializationInfo.javaToJsonFieldNames.get(programField.getName(programClass));
            String   javaFieldName  = programField.getName(programClass);

            if (jsonFieldNames != null)
            {
                // Derive field descriptor and signature (if it exists).
                String                  fieldDescriptor             = programField.getDescriptor(programClass);
                FieldSignatureCollector signatureAttributeCollector = new FieldSignatureCollector();
                programField.attributesAccept(programClass, signatureAttributeCollector);
                boolean retrieveAdapterByTypeToken = false;

                // Check for recursion first if it is an object
                CodeAttributeEditor.Label end = codeAttributeEditor.label();
                if(ClassUtil.isInternalClassType(fieldDescriptor))
                {
                    CodeAttributeEditor.Label noRecursion = codeAttributeEditor.label();
                    ____.aload(OptimizedClassConstants.ToJsonLocals.THIS)
                        .aload(OptimizedClassConstants.ToJsonLocals.THIS)
                        .getfield(programClass, programField)
                        .ifacmpne(noRecursion.offset())
                        .goto_(end.offset())
                        .label(noRecursion);
                }

                if (supportExposeAnnotation &&
                    !classSerializationInfo.exposedJavaFieldNames.contains(javaFieldName))
                {
                    ____.aload(ToJsonLocals.GSON)
                        .getfield(GsonClassConstants.NAME_GSON,     FIELD_NAME_EXCLUDER,       FIELD_TYPE_EXCLUDER)
                        .getfield(GsonClassConstants.NAME_EXCLUDER, FIELD_NAME_REQUIRE_EXPOSE, FIELD_TYPE_REQUIRE_EXPOSE)
                        .ifne(end.offset());
                }


                // Write field name.
                Integer fieldIndex = serializationInfo.jsonFieldIndices.get(jsonFieldNames[0]);
                ____.aload(OptimizedClassConstants.ToJsonLocals.OPTIMIZED_JSON_WRITER)
                    .aload(OptimizedClassConstants.ToJsonLocals.JSON_WRITER)
                    .ldc(fieldIndex.intValue())
                    .invokeinterface(OptimizedClassConstants.NAME_OPTIMIZED_JSON_WRITER,
                                     OptimizedClassConstants.METHOD_NAME_NAME,
                                     OptimizedClassConstants.METHOD_TYPE_NAME);

                // Write field value.
                InlineSerializer inlineSerializer = inlineSerializers.get(fieldDescriptor);
                if (inlineSerializer != null &&
                    inlineSerializer.canSerialize(programClassPool, gsonRuntimeSettings))
                {
                    inlineSerializer.serialize(programClass,
                                               programField,
                                               codeAttributeEditor,
                                               ____,
                                               gsonRuntimeSettings);
                }
                else
                {
                    // Write value to Json writer based on declared type and runtime value/type.
                    ____.aload(OptimizedClassConstants.ToJsonLocals.GSON);

                    switch (fieldDescriptor.charAt(0))
                    {
                        case ClassConstants.TYPE_BOOLEAN:
                        case ClassConstants.TYPE_CHAR:
                        case ClassConstants.TYPE_BYTE:
                        case ClassConstants.TYPE_SHORT:
                        case ClassConstants.TYPE_INT:
                        case ClassConstants.TYPE_FLOAT:
                        case ClassConstants.TYPE_LONG:
                        case ClassConstants.TYPE_DOUBLE:
                        {
                            String className = ClassUtil.internalNumericClassNameFromPrimitiveType(fieldDescriptor.charAt(0));
                            ____.getstatic(className, ClassConstants.FIELD_NAME_TYPE, ClassConstants.FIELD_TYPE_TYPE);
                            break;
                        }
                        case ClassConstants.TYPE_CLASS_START:
                        {
                            if (signatureAttributeCollector.getFieldSignature() == null)
                            {
                                String fieldClassName = fieldDescriptor.substring(1, fieldDescriptor.length() - 1);
                                Clazz  fieldClass     = programClassPool.getClass(fieldClassName);
                                if (fieldClass == null)
                                {
                                    fieldClass = libraryClassPool.getClass(fieldClassName);
                                }
                                ____.ldc(fieldClassName, fieldClass);
                            }
                            else
                            {
                                // Add type token sub-class that has the appropriate type parameter.
                                ProgramClass typeTokenClass =
                                    new TypeTokenClassBuilder(programClass,
                                                              programField,
                                                              signatureAttributeCollector.getFieldSignature())
                                        .build(programClassPool);
                                programClassPool.addClass(typeTokenClass);
                                typeTokenClass.accept(new ClassReferenceInitializer(programClassPool,
                                                                                    libraryClassPool));
                                injectedClassNameMap.put(programClass.getName(), typeTokenClass.getName());

                                // Instantiate type token.
                                ____.new_(typeTokenClass.getName())
                                    .dup()
                                    .invokespecial(typeTokenClass.getName(),
                                                   ClassConstants.METHOD_NAME_INIT,
                                                   ClassConstants.METHOD_TYPE_INIT);
                                retrieveAdapterByTypeToken = true;
                            }
                            break;
                        }
                        case ClassConstants.TYPE_ARRAY:
                        {
                            int fieldDescriptorIndex = 1;
                            while (fieldDescriptor.charAt(fieldDescriptorIndex) == ClassConstants.TYPE_ARRAY)
                            {
                                fieldDescriptorIndex++;
                            }

                            Clazz fieldClass;
                            switch (fieldDescriptor.charAt(fieldDescriptorIndex))
                            {
                                case ClassConstants.TYPE_BOOLEAN:
                                case ClassConstants.TYPE_CHAR:
                                case ClassConstants.TYPE_BYTE:
                                case ClassConstants.TYPE_SHORT:
                                case ClassConstants.TYPE_INT:
                                case ClassConstants.TYPE_FLOAT:
                                case ClassConstants.TYPE_LONG:
                                case ClassConstants.TYPE_DOUBLE:
                                {
                                    String className = ClassUtil.internalNumericClassNameFromPrimitiveType(fieldDescriptor.charAt(0));
                                    fieldClass = libraryClassPool.getClass(className);
                                    ____.ldc(fieldDescriptor, fieldClass);
                                    break;
                                }
                                case ClassConstants.TYPE_CLASS_START:
                                {
                                    String fieldClassName = fieldDescriptor.substring(2, fieldDescriptor.length() - 1);
                                    fieldClass = programClassPool.getClass(fieldClassName);
                                    if (fieldClass == null)
                                    {
                                        fieldClass = libraryClassPool.getClass(fieldClassName);
                                    }
                                    ____.ldc(fieldDescriptor, fieldClass);
                                    break;
                                }
                            }
                            break;
                        }
                    }

                    ____.aload(OptimizedClassConstants.ToJsonLocals.THIS)
                        .getfield(programClass, programField);

                    // Box primitive value before passing it to type adapter.
                    switch (fieldDescriptor.charAt(0))
                    {
                        case ClassConstants.TYPE_BOOLEAN:
                            ____.invokestatic(ClassConstants.NAME_JAVA_LANG_BOOLEAN,
                                              ClassConstants.METHOD_NAME_VALUE_OF,
                                              ClassConstants.METHOD_TYPE_VALUE_OF_BOOLEAN);
                            break;
                        case ClassConstants.TYPE_CHAR:
                            ____.invokestatic(ClassConstants.NAME_JAVA_LANG_CHARACTER,
                                              ClassConstants.METHOD_NAME_VALUE_OF,
                                              ClassConstants.METHOD_TYPE_VALUE_OF_CHAR);
                            break;
                        case ClassConstants.TYPE_BYTE:
                            ____.invokestatic(ClassConstants.NAME_JAVA_LANG_BYTE,
                                              ClassConstants.METHOD_NAME_VALUE_OF,
                                              ClassConstants.METHOD_TYPE_VALUE_OF_BYTE);
                            break;
                        case ClassConstants.TYPE_SHORT:
                            ____.invokestatic(ClassConstants.NAME_JAVA_LANG_SHORT,
                                              ClassConstants.METHOD_NAME_VALUE_OF,
                                              ClassConstants.METHOD_TYPE_VALUE_OF_SHORT);
                            break;
                        case ClassConstants.TYPE_INT:
                            ____.invokestatic(ClassConstants.NAME_JAVA_LANG_INTEGER,
                                              ClassConstants.METHOD_NAME_VALUE_OF,
                                              ClassConstants.METHOD_TYPE_VALUE_OF_INT);
                            break;
                        case ClassConstants.TYPE_FLOAT:
                            ____.invokestatic(ClassConstants.NAME_JAVA_LANG_FLOAT,
                                              ClassConstants.METHOD_NAME_VALUE_OF,
                                              ClassConstants.METHOD_TYPE_VALUE_OF_FLOAT);
                            break;
                        case ClassConstants.TYPE_LONG:
                            ____.invokestatic(ClassConstants.NAME_JAVA_LANG_LONG,
                                              ClassConstants.METHOD_NAME_VALUE_OF,
                                              ClassConstants.METHOD_TYPE_VALUE_OF_LONG);
                            break;
                        case ClassConstants.TYPE_DOUBLE:
                            ____.invokestatic(ClassConstants.NAME_JAVA_LANG_DOUBLE,
                                              ClassConstants.METHOD_NAME_VALUE_OF,
                                              ClassConstants.METHOD_TYPE_VALUE_OF_DOUBLE);
                            break;
                    }

                    // Copy value to local.
                    ____.dup()
                        .astore(valueLocalIndex);

                    // Retrieve type adapter.
                    if(retrieveAdapterByTypeToken)
                    {
                        ____.invokestatic(OptimizedClassConstants.NAME_GSON_UTIL,
                                          OptimizedClassConstants.METHOD_NAME_GET_TYPE_ADAPTER_TYPE_TOKEN,
                                          OptimizedClassConstants.METHOD_TYPE_GET_TYPE_ADAPTER_TYPE_TOKEN);
                    }
                    else
                    {
                        ____.invokestatic(OptimizedClassConstants.NAME_GSON_UTIL,
                                          OptimizedClassConstants.METHOD_NAME_GET_TYPE_ADAPTER_CLASS,
                                          OptimizedClassConstants.METHOD_TYPE_GET_TYPE_ADAPTER_CLASS);
                    }

                    // Write value using type adapter.
                    ____.aload(OptimizedClassConstants.ToJsonLocals.JSON_WRITER)
                        .aload(valueLocalIndex)
                        .invokevirtual(GsonClassConstants.NAME_TYPE_ADAPTER,
                                       GsonClassConstants.METHOD_NAME_WRITE,
                                       GsonClassConstants.METHOD_TYPE_WRITE);
                }

                // Label for skipping writing of field in case of recursion or
                // a non-exposed field with excludeFieldsWithoutExposeAnnotation
                // enabled.
                ____.label(end);
            }
        }
    }
}
