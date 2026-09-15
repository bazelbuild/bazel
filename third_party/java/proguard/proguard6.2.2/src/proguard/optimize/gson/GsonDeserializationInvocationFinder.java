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
import proguard.classfile.visitor.ClassVisitor;
import proguard.evaluation.BasicInvocationUnit;
import proguard.evaluation.value.*;
import proguard.optimize.evaluation.PartialEvaluator;

/**
 * This instruction visitor searches the code for invocations to any of the
 * deserialization methods of Gson (all the fromJson variants) and keeps
 * track of the domain classes that are involved in the deserialization.
 *
 * @author Lars Vandenbergh
 */
public class GsonDeserializationInvocationFinder
extends      SimplifiedVisitor
implements   InstructionVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    public  static       boolean DEBUG = System.getProperty("gdif") != null;
    //*/

    private final ClassPool                   programClassPool;
    private final ClassVisitor                domainClassVisitor;
    private final WarningPrinter              notePrinter;
    private final FromJsonInvocationMatcher[] fromJsonInvocationMatchers;
    private final TypedReferenceValueFactory  valueFactory         =
        new TypedReferenceValueFactory();
    private final PartialEvaluator            partialEvaluator     =
        new PartialEvaluator(valueFactory,
                             new BasicInvocationUnit(new TypedReferenceValueFactory()),
                             true);
    private final AttributeVisitor            lazyPartialEvaluator =
        new AttributeNameFilter(ClassConstants.ATTR_Code,
                                new SingleTimeAttributeVisitor(
                                    partialEvaluator));


    /**
     * Creates a new GsonDeserializationInvocationFinder.
     *
     * @param programClassPool   the program class pool used to look up class
     *                           references.
     * @param domainClassVisitor the visitor to which found domain classes that
     *                           are involved in Gson deserialization will
     *                           be delegated.
     * @param notePrinter        used to print notes about domain classes that
     *                           can not be handled by the Gson optimization.
     */
    public GsonDeserializationInvocationFinder(ClassPool      programClassPool,
                                               ClassVisitor   domainClassVisitor,
                                               WarningPrinter notePrinter)
    {
        this.programClassPool   = programClassPool;
        this.domainClassVisitor = domainClassVisitor;
        this.notePrinter        = notePrinter;

        // Create matchers for relevant instruction sequences.
        InstructionSequenceBuilder builder = new InstructionSequenceBuilder();

        // The invocation "Gson#fromJson(String, Class<T>)".
        Instruction[] fromJsonStringClassInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON,
                           GsonClassConstants.METHOD_NAME_FROM_JSON,
                           GsonClassConstants.METHOD_TYPE_FROM_JSON_STRING_CLASS)
            .instructions();

        // The invocation "Gson#fromJson(String, Type)".
        Instruction[] fromJsonStringTypeInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON,
                           GsonClassConstants.METHOD_NAME_FROM_JSON,
                           GsonClassConstants.METHOD_TYPE_FROM_JSON_STRING_TYPE)
            .instructions();

        // The invocation "Gson#fromJson(Reader, Class<T>)".
        Instruction[] fromJsonReaderClassInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON,
                           GsonClassConstants.METHOD_NAME_FROM_JSON,
                           GsonClassConstants.METHOD_TYPE_FROM_JSON_READER_CLASS)
            .instructions();

        // The invocation "Gson#fromJson(Reader, Type)".
        Instruction[] fromJsonReaderTypeInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON,
                           GsonClassConstants.METHOD_NAME_FROM_JSON,
                           GsonClassConstants.METHOD_TYPE_FROM_JSON_READER_TYPE)
            .instructions();

        // The invocation "Gson#fromJson(JsonReader, Type)".
        Instruction[] fromJsonJsonReaderTypeInstructions = builder
            .invokevirtual(GsonClassConstants.NAME_GSON,
                           GsonClassConstants.METHOD_NAME_FROM_JSON,
                           GsonClassConstants.METHOD_TYPE_FROM_JSON_JSON_READER_TYPE)
            .instructions();

        Constant[] constants = builder.constants();

        fromJsonInvocationMatchers = new FromJsonInvocationMatcher[]
        {
            new FromJsonInvocationMatcher(constants, fromJsonStringClassInstructions,      0, -1),
            new FromJsonInvocationMatcher(constants, fromJsonStringTypeInstructions,      -1,  0),
            new FromJsonInvocationMatcher(constants, fromJsonReaderClassInstructions,      0, -1),
            new FromJsonInvocationMatcher(constants, fromJsonReaderTypeInstructions,      -1,  0),
            new FromJsonInvocationMatcher(constants, fromJsonJsonReaderTypeInstructions,  -1,  0)
        };
    }


    // Implementations for InstructionVisitor.

    @Override
    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        // Try to match any of the fromJson() constructs.
        FromJsonInvocationMatcher matchingMatcher = null;
        for (FromJsonInvocationMatcher matcher : fromJsonInvocationMatchers)
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
                System.out.println("GsonDeserializationInvocationFinder: Gson#fromJson: " +
                                   clazz.getName() +
                                   "." +
                                   method.getName(clazz) +
                                   method.getDescriptor(clazz) +
                                   " " +
                                   instruction.toString(offset));
            }

            // Figure out the type that is being deserialized.
            lazyPartialEvaluator.visitCodeAttribute(clazz,
                                                    method,
                                                    codeAttribute);

            // Derive types from Class or Type argument.
            int stackElementIndex = matchingMatcher.typeStackElementIndex == -1 ?
                matchingMatcher.classStackElementIndex :
                matchingMatcher.typeStackElementIndex;
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
                                     "Note: can't derive deserialized type from fromJson() invocation in " +
                                     clazz.getName() +
                                     "." +
                                     method.getName(clazz) +
                                     method.getDescriptor(clazz));
            }
        }
    }


    // Utility classes.

    private static class FromJsonInvocationMatcher
    extends              InstructionSequenceMatcher
    {
        private int classStackElementIndex;
        private int typeStackElementIndex;

        private FromJsonInvocationMatcher(Constant[]    patternConstants,
                                          Instruction[] patternInstructions,
                                          int           classStackElementIndex,
                                          int           typeStackElementIndex)
        {
            super(patternConstants, patternInstructions);
            this.classStackElementIndex = classStackElementIndex;
            this.typeStackElementIndex  = typeStackElementIndex;
        }
    }
}
