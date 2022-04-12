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
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.Constant;
import proguard.classfile.editor.InstructionSequenceBuilder;
import proguard.classfile.instruction.Instruction;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.evaluation.BasicInvocationUnit;
import proguard.evaluation.value.*;
import proguard.optimize.evaluation.PartialEvaluator;

/**
 * This instructor visitor searches for invocations to GsonBuilder and keeps
 * track of which parameters of the GsonBuilder are being utilized in the code
 * in a GsonRuntimeSettings instance.
 *
 * @author Joachim Vandersmissen
 * @author Lars Vandenbergh
 */
public class GsonBuilderInvocationFinder
extends      SimplifiedVisitor
implements   InstructionVisitor
{
    private final InstructionSequenceMatcher setVersionMatcher;
    private final InstructionSequenceMatcher excludeFieldsWithModifiersMatcher;
    private final InstructionSequenceMatcher generateNonExecutableJsonMatcher;
    private final InstructionSequenceMatcher excludeFieldsWithoutExposeAnnotationMatcher;
    private final InstructionSequenceMatcher serializeNullsMatcher;
    private final InstructionSequenceMatcher disableInnerClassSerializationMatcher;
    private final InstructionSequenceMatcher setLongSerializationPolicyMatcher;
    private final InstructionSequenceMatcher setFieldNamingPolicyMatcher;
    private final InstructionSequenceMatcher setFieldNamingStrategyMatcher;
    private final InstructionSequenceMatcher setExclusionStrategiesMatcher;
    private final InstructionSequenceMatcher addSerializationExclusionStrategyMatcher;
    private final InstructionSequenceMatcher addDeserializationExclusionStrategyMatcher;
    private final InstructionSequenceMatcher registerTypeAdapterMatcher;
    private final InstructionSequenceMatcher registerTypeHierachyAdapterMatcher;
    private final InstructionSequenceMatcher serializeSpecialFloatingPointValuesMatcher;
    private final TypedReferenceValueFactory valueFactory         =
        new TypedReferenceValueFactory();
    private final PartialEvaluator           partialEvaluator     =
        new PartialEvaluator(valueFactory,
                             new BasicInvocationUnit(new TypedReferenceValueFactory()),
                             true);
    private final AttributeVisitor           lazyPartialEvaluator =
        new AttributeNameFilter(ClassConstants.ATTR_Code,
                                new SingleTimeAttributeVisitor(
                                             partialEvaluator));
    private final ClassPool                  programClassPool;
    private final GsonRuntimeSettings        gsonRuntimeSettings;
    private final ClassVisitor               instanceCreatorClassVisitor;
    private final ClassVisitor               typeAdapterClassVisitor;

    /**
     * Creates a new GsonBuilderInvocationFinder.
     *
     * @param programClassPool            the program class pool used to look
     *                                    up class references.
     * @param instanceCreatorClassVisitor visitor to which domain classes for
     *                                    which an InstanceCreator is
     *                                    registered will be delegated.
     * @param typeAdapterClassVisitor     visitor to which domain classes for
     *                                    which a TypeAdapter is registered
     *                                    will be delegated.
     */
    public GsonBuilderInvocationFinder(ClassPool           programClassPool,
                                       GsonRuntimeSettings gsonRuntimeSettings,
                                       ClassVisitor        instanceCreatorClassVisitor,
                                       ClassVisitor        typeAdapterClassVisitor)
    {
        this.programClassPool            = programClassPool;
        this.gsonRuntimeSettings         = gsonRuntimeSettings;
        this.instanceCreatorClassVisitor = instanceCreatorClassVisitor;
        this.typeAdapterClassVisitor     = typeAdapterClassVisitor;

        InstructionSequenceBuilder builder = new InstructionSequenceBuilder();

        Instruction[] setVersionInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_SET_VERSION,
                           GsonClassConstants.METHOD_TYPE_SET_VERSION)
            .instructions();

        Instruction[] excludeFieldsWithModifiersInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_EXCLUDE_FIELDS_WITH_MODIFIERS,
                           GsonClassConstants.METHOD_TYPE_EXCLUDE_FIELDS_WITH_MODIFIERS)
            .instructions();

        Instruction[] generateNonExecutableJsonInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_GENERATE_EXECUTABLE_JSON,
                           GsonClassConstants.METHOD_TYPE_GENERATE_EXECUTABLE_JSON)
            .instructions();

        Instruction[] excludeFieldsWithoutExposeAnnotationInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_EXCLUDE_FIELDS_WITHOUT_EXPOSE_ANNOTATION,
                           GsonClassConstants.METHOD_TYPE_EXLCUDE_FIELDS_WITHOUT_EXPOSE_ANNOTATION)
            .instructions();

        Instruction[] serializeNullsInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_SERIALIZE_NULLS,
                           GsonClassConstants.METHOD_TYPE_SERIALIZE_NULLS)
            .instructions();

        Instruction[] enableComplexMapKeySerializationInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_ENABLE_COMPLEX_MAP_KEY_SERIALIZATION,
                           GsonClassConstants.METHOD_TYPE_ENABLE_COMPLEX_MAP_KEY_SERIALIZATION)
            .instructions();

        Instruction[] disableInnerClassSerializationInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_DISABLE_INNER_CLASS_SERIALIZATION,
                           GsonClassConstants.METHOD_TYPE_DISABLE_INNER_CLASS_SERIALIZATION)
            .instructions();

        Instruction[] setLongSerializationPolicyInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_SET_LONG_SERIALIZATION_POLICY,
                           GsonClassConstants.METHOD_TYPE_SET_LONG_SERIALIZATION_POLICY)
            .instructions();

        Instruction[] setFieldNamingStrategyInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_SET_FIELD_NAMING_STRATEGY,
                           GsonClassConstants.METHOD_TYPE_SET_FIELD_NAMING_STRATEGY)
            .instructions();

        Instruction[] setFieldNamingPolicyInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_SET_FIELD_NAMING_POLICY,
                           GsonClassConstants.METHOD_TYPE_SET_FIELD_NAMING_POLICY)
            .instructions();

        Instruction[] setExclusionStrategiesInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_SET_EXCLUSION_STRATEGIES,
                           GsonClassConstants.METHOD_TYPE_SET_EXCLUSION_STRATEGIES)
            .instructions();

        Instruction[] addSerializationExclusionStrategyInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_ADD_SERIALIZATION_EXCLUSION_STRATEGY,
                           GsonClassConstants.METHOD_TYPE_ADD_SERIALIZATION_EXCLUSION_STRATEGY)
            .instructions();

        Instruction[] addDeserializationExclusionStrategyInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_ADD_DESERIALIZATION_EXCLUSION_STRATEGY,
                           GsonClassConstants.METHOD_TYPE_ADD_DESERIALIZATION_EXCLUSION_STRATEGY)
            .instructions();

        Instruction[] registerTypeAdapterInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_REGISTER_TYPE_ADAPTER,
                           GsonClassConstants.METHOD_TYPE_REGISTER_TYPE_ADAPTER)
            .instructions();

        Instruction[] registerTypeHierarchyAdapterInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_REGISTER_TYPE_HIERARCHY_ADAPTER,
                           GsonClassConstants.METHOD_TYPE_REGISTER_TYPE_HIERARCHY_ADAPTER)
            .instructions();

        Instruction[] serializeSpecialFloatingPointValuesInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON_BUILDER,
                           GsonClassConstants.METHOD_NAME_SERIALIZE_SPECIAL_FLOATING_POINT_VALUES,
                           GsonClassConstants.METHOD_TYPE_SERIALIZE_SPECIAL_FLOATING_POINT_VALUES)
            .instructions();

        Constant[] constants = builder.constants();

        setVersionMatcher = new InstructionSequenceMatcher(constants,
                                                           setVersionInstructions);

        excludeFieldsWithModifiersMatcher =
            new InstructionSequenceMatcher(constants,
                                           excludeFieldsWithModifiersInstructions);

        generateNonExecutableJsonMatcher =
            new InstructionSequenceMatcher(constants,
                                           generateNonExecutableJsonInstructions);

        excludeFieldsWithoutExposeAnnotationMatcher =
            new InstructionSequenceMatcher(constants,
                                           excludeFieldsWithoutExposeAnnotationInstructions);

        serializeNullsMatcher = new InstructionSequenceMatcher(constants,
                                                               serializeNullsInstructions);

        disableInnerClassSerializationMatcher =
            new InstructionSequenceMatcher(constants,
                                           disableInnerClassSerializationInstructions);

        setLongSerializationPolicyMatcher =
            new InstructionSequenceMatcher(constants,
                                           setLongSerializationPolicyInstructions);

        setFieldNamingPolicyMatcher =
            new InstructionSequenceMatcher(constants,
                                           setFieldNamingPolicyInstructions);

        setFieldNamingStrategyMatcher =
            new InstructionSequenceMatcher(constants,
                                           setFieldNamingStrategyInstructions);

        setExclusionStrategiesMatcher =
            new InstructionSequenceMatcher(constants,
                                           setExclusionStrategiesInstructions);

        addSerializationExclusionStrategyMatcher =
            new InstructionSequenceMatcher(constants,
                                           addSerializationExclusionStrategyInstructions);

        addDeserializationExclusionStrategyMatcher =
            new InstructionSequenceMatcher(constants,
                                           addDeserializationExclusionStrategyInstructions);

        registerTypeAdapterMatcher =
            new InstructionSequenceMatcher(constants,
                                           registerTypeAdapterInstructions);

        registerTypeHierachyAdapterMatcher =
            new InstructionSequenceMatcher(constants,
                                           registerTypeHierarchyAdapterInstructions);

        serializeSpecialFloatingPointValuesMatcher =
            new InstructionSequenceMatcher(constants,
                                           serializeSpecialFloatingPointValuesInstructions);
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz         clazz,
                                    Method        method,
                                    CodeAttribute codeAttribute,
                                    int           offset,
                                    Instruction   instruction)
    {
        if (!gsonRuntimeSettings.setVersion)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               setVersionMatcher);

            gsonRuntimeSettings.setVersion = setVersionMatcher.isMatching();
        }

        if (!gsonRuntimeSettings.excludeFieldsWithModifiers)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               excludeFieldsWithModifiersMatcher);

            gsonRuntimeSettings.excludeFieldsWithModifiers =
                excludeFieldsWithModifiersMatcher.isMatching();
        }

        if (!gsonRuntimeSettings.generateNonExecutableJson)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               generateNonExecutableJsonMatcher);

            gsonRuntimeSettings.generateNonExecutableJson =
                generateNonExecutableJsonMatcher.isMatching();
        }

        if (!gsonRuntimeSettings.excludeFieldsWithoutExposeAnnotation)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               excludeFieldsWithoutExposeAnnotationMatcher);

            gsonRuntimeSettings.excludeFieldsWithoutExposeAnnotation =
                excludeFieldsWithoutExposeAnnotationMatcher.isMatching();
        }

        if (!gsonRuntimeSettings.serializeNulls)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               serializeNullsMatcher);

            gsonRuntimeSettings.serializeNulls =
                serializeNullsMatcher.isMatching();
        }

        if (!gsonRuntimeSettings.disableInnerClassSerialization)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               disableInnerClassSerializationMatcher);

            gsonRuntimeSettings.disableInnerClassSerialization =
                disableInnerClassSerializationMatcher.isMatching();
        }

        if (!gsonRuntimeSettings.setLongSerializationPolicy)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               setLongSerializationPolicyMatcher);

            gsonRuntimeSettings.setLongSerializationPolicy =
                setLongSerializationPolicyMatcher.isMatching();
        }

        if (!gsonRuntimeSettings.setFieldNamingPolicy)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               setFieldNamingPolicyMatcher);

            gsonRuntimeSettings.setFieldNamingPolicy =
                setFieldNamingPolicyMatcher.isMatching();
        }

        if (!gsonRuntimeSettings.setFieldNamingStrategy)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               setFieldNamingStrategyMatcher);

            gsonRuntimeSettings.setFieldNamingStrategy =
                setFieldNamingStrategyMatcher.isMatching();
        }

        if (!gsonRuntimeSettings.setExclusionStrategies)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               setExclusionStrategiesMatcher);

            gsonRuntimeSettings.setExclusionStrategies =
                setExclusionStrategiesMatcher.isMatching();
        }

        if (!gsonRuntimeSettings.addSerializationExclusionStrategy)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               addSerializationExclusionStrategyMatcher);

            gsonRuntimeSettings.addSerializationExclusionStrategy =
                addSerializationExclusionStrategyMatcher.isMatching();
        }

        if (!gsonRuntimeSettings.addDeserializationExclusionStrategy)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               addDeserializationExclusionStrategyMatcher);

            gsonRuntimeSettings.addDeserializationExclusionStrategy =
                addDeserializationExclusionStrategyMatcher.isMatching();
        }

        if (!gsonRuntimeSettings.serializeSpecialFloatingPointValues)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               serializeSpecialFloatingPointValuesMatcher);

            gsonRuntimeSettings.serializeSpecialFloatingPointValues =
                serializeSpecialFloatingPointValuesMatcher.isMatching();
        }

        if (instanceCreatorClassVisitor != null && typeAdapterClassVisitor != null)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               registerTypeAdapterMatcher);
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               registerTypeHierachyAdapterMatcher);

            if (registerTypeAdapterMatcher.isMatching() ||
                registerTypeHierachyAdapterMatcher.isMatching())
            {
                // Figure out the class for which a type adapter is registered.
                lazyPartialEvaluator.visitCodeAttribute(clazz,
                                                        method,
                                                        codeAttribute);

                // Derive Class from type argument.
                InstructionOffsetValue typeProducer =
                    partialEvaluator.getStackBefore(offset)
                        .getTopActualProducerValue(1)
                        .instructionOffsetValue();

                TypeArgumentFinder typeArgumentFinder =
                    new TypeArgumentFinder(programClassPool, partialEvaluator);
                for (int i = 0; i < typeProducer.instructionOffsetCount(); i++)
                {
                    codeAttribute.instructionAccept(clazz,
                                                    method,
                                                    typeProducer.instructionOffset(i),
                                                    typeArgumentFinder);
                }

                if (typeArgumentFinder.typeArgumentClasses != null &&
                    typeArgumentFinder.typeArgumentClasses.length == 1)
                {
                    String typeArgumentClass =
                        typeArgumentFinder.typeArgumentClasses[0];
                    Clazz type = programClassPool.getClass(typeArgumentClass);

                    if (type != null)
                    {
                        // Derive Class from typeAdapter argument.
                        InstructionOffsetValue typeAdapterProducer =
                            partialEvaluator.getStackBefore(offset)
                                .getTopActualProducerValue(0)
                                .instructionOffsetValue();

                        TypeArgumentFinder typeAdapterArgumentFinder =
                            new TypeArgumentFinder(programClassPool, partialEvaluator);
                        for (int i = 0; i < typeAdapterProducer.instructionOffsetCount(); i++)
                        {
                            codeAttribute.instructionAccept(clazz,
                                                            method,
                                                            typeAdapterProducer.instructionOffset(i),
                                                            typeAdapterArgumentFinder);
                        }

                        if (typeAdapterArgumentFinder.typeArgumentClasses != null &&
                            typeAdapterArgumentFinder.typeArgumentClasses.length == 1)
                        {
                            // Check whether type adapter passed as argument
                            // implements InstanceCreator before passing the
                            // domain type itself to the instanceCreatorClassVisitor.
                            String typeAdapterArgumentClass =
                                typeAdapterArgumentFinder.typeArgumentClasses[0];
                            Clazz instanceCreator =
                                programClassPool.getClass(GsonClassConstants.NAME_INSTANCE_CREATOR);
                            ImplementedClassFilter implementsInstanceCreatorFilter =
                                new ImplementedClassFilter(instanceCreator,
                                                           false,
                                                           new ClassVisitorPropagator(type, instanceCreatorClassVisitor),
                                                           new ClassVisitorPropagator(type, typeAdapterClassVisitor));
                            programClassPool.classAccept(typeAdapterArgumentClass, implementsInstanceCreatorFilter);
                        }
                    }
                }
            }
        }
    }

    private static class ClassVisitorPropagator
    implements           ClassVisitor
    {
        private final Clazz        clazz;
        private final ClassVisitor classVisitor;

        private ClassVisitorPropagator(Clazz        clazz,
                                       ClassVisitor classVisitor)
        {
            this.clazz        = clazz;
            this.classVisitor = classVisitor;
        }

        // Implementations for ClassVisitor

        @Override
        public void visitProgramClass(ProgramClass programClass)
        {
            clazz.accept(classVisitor);
        }


        @Override
        public void visitLibraryClass(LibraryClass libraryClass) {}
    }
}
