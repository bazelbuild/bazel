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
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.evaluation.TracedStack;
import proguard.evaluation.value.InstructionOffsetValue;
import proguard.optimize.evaluation.PartialEvaluator;
import proguard.util.ArrayUtil;

/**
 * This instruction visitor searches for the types that are passed as arguments
 * to a method. The visited instruction is assumed to be the producer that put
 * the types on the stack.
 *
 * It can recognize Class objects and Type objects that were derived from
 * TypeToken.getType().
 *
 * @author Lars Vandenbergh
 */
class      TypeArgumentFinder
extends    SimplifiedVisitor
implements InstructionVisitor,
           ConstantVisitor
{
    private final ClassPool        programClassPool;
    private final PartialEvaluator partialEvaluator;
                  String[]         typeArgumentClasses;


    /**
     * Creates a new TypeArgumentFinder.
     *
     * @param programClassPool the program class pool used for looking up
     *                         class references.
     * @param partialEvaluator the partial evaluator used to evaluate visited
     *                         code attributes.
     */
    TypeArgumentFinder(ClassPool        programClassPool,
                       PartialEvaluator partialEvaluator)
    {
        this.programClassPool = programClassPool;
        this.partialEvaluator = partialEvaluator;
    }


    // Implementations for InstructionVisitor.

    @Override
    public void visitAnyInstruction(Clazz         clazz,
                                    Method        method,
                                    CodeAttribute codeAttribute,
                                    int           offset,
                                    Instruction   instruction) {}

    @Override
    public void visitVariableInstruction(Clazz               clazz,
                                         Method              method,
                                         CodeAttribute       codeAttribute,
                                         int                 offset,
                                         VariableInstruction variableInstruction)
    {
        if (variableInstruction.canonicalOpcode() == InstructionConstants.OP_ALOAD)
        {
            // Find the operation that stored the loaded Type.
            LastStoreFinder lastStoreFinder = new LastStoreFinder(variableInstruction.variableIndex);
            codeAttribute.instructionsAccept(clazz, method, 0, offset, lastStoreFinder);

            if (lastStoreFinder.lastStore != null)
            {
                // Find out which instruction produced the stored Type.
                TracedStack stackBeforeStore = partialEvaluator.getStackBefore(lastStoreFinder.lastStoreOffset);
                InstructionOffsetValue instructionOffsetValue = stackBeforeStore.getTopProducerValue(0).instructionOffsetValue();

                // Derive the signature of the subclass of TypeToken from which the Type is retrieved.
                TypeTokenSignatureFinder typeTokenFinder = new TypeTokenSignatureFinder();
                for (int offsetIndex = 0; offsetIndex < instructionOffsetValue.instructionOffsetCount(); offsetIndex++)
                {
                    int instructionOffset = instructionOffsetValue.instructionOffset(offsetIndex);
                    codeAttribute.instructionAccept(clazz, method, instructionOffset, typeTokenFinder);
                }

                // Derive the classes from the signature of the TypeToken subclass.
                if (typeTokenFinder.typeTokenSignature != null)
                {
                    typeArgumentClasses = new String[0];
                    Clazz[] referencedClasses = typeTokenFinder.typeTokenSignature.referencedClasses;
                    for (Clazz referencedClass : referencedClasses)
                    {
                        if (referencedClass!= null &&
                            !referencedClass.getName().equals(GsonClassConstants.NAME_TYPE_TOKEN))
                        {
                            typeArgumentClasses = ArrayUtil.add(typeArgumentClasses,
                                                                typeArgumentClasses.length,
                                                                referencedClass.getName());
                        }
                    }
                }
            }
        }
    }


    public void visitConstantInstruction(Clazz               clazz,
                                         Method              method,
                                         CodeAttribute       codeAttribute,
                                         int                 offset,
                                         ConstantInstruction constantInstruction)
    {
        clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
    }

    // Implementations for ConstantVisitor.


    @Override
    public void visitAnyConstant(Clazz clazz, Constant constant)
    {
    }

    @Override
    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        typeArgumentClasses = new String[] { refConstant.getClassName(clazz) };
    }

    @Override
    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        typeArgumentClasses = new String[] { classConstant.getName(clazz) };
    }

    private static class LastStoreFinder
    extends              SimplifiedVisitor
    implements           InstructionVisitor
    {
        private final int           variableIndex;
        private int                 lastStoreOffset;
        private VariableInstruction lastStore;

        public LastStoreFinder(int variableIndex)
        {
            this.variableIndex = variableIndex;
        }

        // Implementations for InstructionVisitor.

        @Override
        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
        {
        }

        @Override
        public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
        {
            if(variableInstruction.variableIndex == variableIndex &&
               variableInstruction.canonicalOpcode() == InstructionConstants.OP_ASTORE){
                lastStoreOffset = offset;
                lastStore = variableInstruction;
            }
        }
    }

    private class TypeTokenSignatureFinder
    extends       SimplifiedVisitor
    implements    InstructionVisitor,
                  ConstantVisitor,
                  AttributeVisitor
    {

        private SignatureAttribute typeTokenSignature;

        // Implementations for InstructionVisitor.

        @Override
        public void visitAnyInstruction(Clazz         clazz,
                                        Method        method,
                                        CodeAttribute codeAttribute,
                                        int           offset,
                                        Instruction   instruction)
        {
        }

        @Override
        public void visitConstantInstruction(Clazz               clazz,
                                             Method              method,
                                             CodeAttribute       codeAttribute,
                                             int                 offset,
                                             ConstantInstruction constantInstruction)
        {
            clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
        }

        // Implementations for ConstantVisitor.

        @Override
        public void visitAnyConstant(Clazz clazz, Constant constant)
        {
        }

        @Override
        public void visitMethodrefConstant(Clazz             clazz,
                                           MethodrefConstant methodrefConstant)
        {
            if (methodrefConstant.referencedClass.getName().equals(GsonClassConstants.NAME_TYPE_TOKEN) &&
                methodrefConstant.getName(clazz).equals(GsonClassConstants.METHOD_NAME_GET_TYPE))
            {
                programClassPool.classAccept(methodrefConstant.getClassName(clazz),
                                             new AllAttributeVisitor(this));
            }
        }

        // Implementations for AttributeVisitor.

        @Override
        public void visitAnyAttribute(Clazz clazz, Attribute attribute)
        {
        }

        @Override
        public void visitSignatureAttribute(Clazz              clazz,
                                            SignatureAttribute signatureAttribute)
        {
            typeTokenSignature = signatureAttribute;
        }
    }
}
