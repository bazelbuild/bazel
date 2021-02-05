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
 * This instruction visitor searches the code for invocations to any of the
 * serialization methods of Gson (all the toJson variants) and keeps
 * track of the domain classes that are involved in the serialization.
 *
 * @author Lars Vandenbergh
 */
public class GsonSerializationInvocationFinder
extends      SimplifiedVisitor
implements   InstructionVisitor
{
    private static final boolean DEBUG = false;

    private final ClassPool                  programClassPool;
    private final ClassVisitor               domainClassVisitor;
    private final WarningPrinter             notePrinter;
    private final ToJsonInvocationMatcher[]  toJsonInvocationMatchers;
    private final TypedReferenceValueFactory valueFactory         =
        new TypedReferenceValueFactory();
    private final PartialEvaluator           partialEvaluator     =
        new PartialEvaluator(valueFactory,
                             new BasicInvocationUnit(new TypedReferenceValueFactory()),
                             true);
    private final AttributeVisitor           lazyPartialEvaluator =
        new AttributeNameFilter(ClassConstants.ATTR_Code,
                                new SingleTimeAttributeVisitor(partialEvaluator));


    /**
     * Creates a new GsonSerializationInvocationFinder.
     *
     * @param programClassPool   the program class pool used to look up class
     *                           references.
     * @param domainClassVisitor the visitor to which found domain classes that
     *                           are involved in Gson serialization will
     *                           be delegated.
     * @param notePrinter        used to print notes about domain classes that
     *                           can not be handled by the Gson optimization.
     */
    public GsonSerializationInvocationFinder(ClassPool      programClassPool,
                                             ClassVisitor   domainClassVisitor,
                                             WarningPrinter notePrinter)
    {
        this.programClassPool   = programClassPool;
        this.domainClassVisitor = domainClassVisitor;
        this.notePrinter        = notePrinter;

        // Create matchers for relevant instruction sequences.
        InstructionSequenceBuilder builder = new InstructionSequenceBuilder();

        // The invocation "Gson#toJson(Object)".
        Instruction[] toJsonObjectInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON,
                           GsonClassConstants.METHOD_NAME_TO_JSON,
                           GsonClassConstants.METHOD_TYPE_TO_JSON_OBJECT)
            .instructions();

        // The invocation "Gson#toJson(Object, Type)".
        Instruction[] toJsonObjectTypeInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON,
                           GsonClassConstants.METHOD_NAME_TO_JSON,
                           GsonClassConstants.METHOD_TYPE_TO_JSON_OBJECT_TYPE)
            .instructions();

        // The invocation "Gson#toJson(Object, Appendable)".
        Instruction[] toJsonObjectAppendableInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON,
                           GsonClassConstants.METHOD_NAME_TO_JSON,
                           GsonClassConstants.METHOD_TYPE_TO_JSON_OBJECT_APPENDABLE)
            .instructions();

        // The invocation "Gson#toJson(Object, Type, Appendable)".
        Instruction[] toJsonObjectTypeAppendableInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON,
                           GsonClassConstants.METHOD_NAME_TO_JSON,
                           GsonClassConstants.METHOD_TYPE_TO_JSON_OBJECT_TYPE_APPENDABLE)
            .instructions();

        // The invocation "Gson#toJson(Object, Type, JsonWriter)".
        Instruction[] toJsonObjectTypeWriterInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON,
                           GsonClassConstants.METHOD_NAME_TO_JSON,
                           GsonClassConstants.METHOD_TYPE_TO_JSON_OBJECT_TYPE_WRITER)
            .instructions();

        // The invocation "Gson#toJsonTree(Object)".
        Instruction[] toJsonTreeObjectInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON,
                           GsonClassConstants.METHOD_NAME_TO_JSON_TREE,
                           GsonClassConstants.METHOD_TYPE_TO_JSON_TREE_OBJECT)
            .instructions();

        // The invocation "Gson#toJsonTree(Object, Type)".
        Instruction[] toJsonTreeObjectTypeInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON,
                           GsonClassConstants.METHOD_NAME_TO_JSON_TREE,
                           GsonClassConstants.METHOD_TYPE_TO_JSON_TREE_OBJECT_TYPE)
            .instructions();

        Constant[] constants = builder.constants();

        toJsonInvocationMatchers = new ToJsonInvocationMatcher[]
        {
            new ToJsonInvocationMatcher(constants, toJsonObjectInstructions              , 0, -1),
            new ToJsonInvocationMatcher(constants, toJsonObjectTypeInstructions          , 1,  0),
            new ToJsonInvocationMatcher(constants, toJsonObjectAppendableInstructions    , 1, -1),
            new ToJsonInvocationMatcher(constants, toJsonObjectTypeAppendableInstructions, 2,  1),
            new ToJsonInvocationMatcher(constants, toJsonObjectTypeWriterInstructions    , 2,  1),
            new ToJsonInvocationMatcher(constants, toJsonTreeObjectInstructions          , 0, -1),
            new ToJsonInvocationMatcher(constants, toJsonTreeObjectTypeInstructions      , 1,  0)
        };
    }


    // Implementations for InstructionVisitor.

    @Override
    public void visitAnyInstruction(Clazz         clazz,
                                    Method        method,
                                    CodeAttribute codeAttribute,
                                    int           offset,
                                    Instruction   instruction)
    {
        // Try to match any of the toJson() constructs.
        ToJsonInvocationMatcher matchingMatcher = null;
        for (ToJsonInvocationMatcher matcher : toJsonInvocationMatchers)
        {
            instruction.accept(clazz,
                               method,
                               codeAttribute,
                               offset,
                               matcher);
            if(matcher.isMatching())
            {
                matchingMatcher = matcher;
                break;
            }
        }

        if (matchingMatcher != null)
        {
            if (DEBUG)
            {
                System.out.println("GsonSerializationInvocationFinder: Gson#toJson/toJsonTree: " +
                                   clazz.getName() +
                                   "." +
                                   method.getName(clazz) +
                                   method.getDescriptor(clazz) +
                                   " " +
                                   instruction.toString(offset));
            }

            // Figure out the type that is being serialized.
            lazyPartialEvaluator.visitCodeAttribute(clazz,
                                                    method,
                                                    codeAttribute);

            if (matchingMatcher.typeStackElementIndex == -1)
            {
                // Derive type from Object argument.
                int stackElementIndex = matchingMatcher.objectStackElementIndex;
                ReferenceValue top = partialEvaluator.getStackBefore(offset)
                                                     .getTop(stackElementIndex)
                                                     .referenceValue();
                Clazz targetClass = top.getReferencedClass();

                if (targetClass instanceof ProgramClass)
                {
                    targetClass.accept(domainClassVisitor);
                }
            }
            else
            {
                // Derive types from Type argument.
                int stackElementIndex = matchingMatcher.typeStackElementIndex;
                InstructionOffsetValue producer =
                    partialEvaluator.getStackBefore(offset)
                                    .getTopActualProducerValue(stackElementIndex)
                                    .instructionOffsetValue();

                TypeArgumentFinder typeArgumentFinder =
                    new TypeArgumentFinder(programClassPool, partialEvaluator);
                for (int i = 0; i < producer.instructionOffsetCount(); i++)
                {
                    codeAttribute.instructionAccept(clazz,
                                                    method,
                                                    producer.instructionOffset(i),
                                                    typeArgumentFinder);
                }

                String[] targetTypes = typeArgumentFinder.typeArgumentClasses;
                if (targetTypes != null)
                {
                    for (String targetType : targetTypes)
                    {
                        programClassPool.classAccept(targetType, domainClassVisitor);
                    }
                }
                else if (notePrinter != null)
                {
                    notePrinter.print(clazz.getName(),
                                      "Warning: can't derive serialized type from toJson() invocation in " +
                                      clazz.getName() +
                                      "." +
                                      method.getName(clazz) +
                                      method.getDescriptor(clazz));
                }
            }
        }
    }


    // Utility classes.

    private static class ToJsonInvocationMatcher
    extends              InstructionSequenceMatcher
    {
        private int objectStackElementIndex;
        private int typeStackElementIndex;

        private ToJsonInvocationMatcher(Constant[]   patternConstants,
                                       Instruction[] patternInstructions,
                                       int           objectStackElementIndex,
                                       int           typeStackElementIndex)
        {
            super(patternConstants, patternInstructions);
            this.objectStackElementIndex = objectStackElementIndex;
            this.typeStackElementIndex   = typeStackElementIndex;
        }
    }
}
