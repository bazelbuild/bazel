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

package proguard.optimize.info;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.evaluation.*;
import proguard.evaluation.value.*;
import proguard.optimize.evaluation.*;
import proguard.util.ArrayUtil;

/**
 * This AttributeVisitor can tell whether reference parameters and instances
 * are escaping, are modified, or are returned.
 *
 * @see ParameterEscapeMarker
 * @author Eric Lafortune
 */
public class ReferenceEscapeChecker
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor,
             ConstantVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("rec") != null;
    //*/


    private boolean[] instanceEscaping  = new boolean[ClassConstants.TYPICAL_CODE_LENGTH];
    private boolean[] instanceReturned  = new boolean[ClassConstants.TYPICAL_CODE_LENGTH];
    private boolean[] instanceModified  = new boolean[ClassConstants.TYPICAL_CODE_LENGTH];
    private boolean[] externalInstance  = new boolean[ClassConstants.TYPICAL_CODE_LENGTH];
//    private boolean[] exceptionEscaping = new boolean[ClassConstants.TYPICAL_CODE_LENGTH];
//    private boolean[] exceptionReturned = new boolean[ClassConstants.TYPICAL_CODE_LENGTH];
//    private boolean[] exceptionModified = new boolean[ClassConstants.TYPICAL_CODE_LENGTH];

    private final PartialEvaluator partialEvaluator;
    private final boolean          runPartialEvaluator;

    // Parameters and values for visitor methods.
    private Method referencingMethod;
    private int    referencingOffset;
    private int    referencingPopCount;


    /**
     * Creates a new ReferenceEscapeChecker.
     */
    public ReferenceEscapeChecker()
    {
        this(new ReferenceTracingValueFactory(new BasicValueFactory()));
    }


    /**
     * Creates a new ReferenceEscapeChecker. This private constructor gets around
     * the constraint that it's not allowed to add statements before calling
     * 'this'.
     */
    private ReferenceEscapeChecker(ReferenceTracingValueFactory referenceTracingValueFactory)
    {
        this(new PartialEvaluator(referenceTracingValueFactory,
                                  new ParameterTracingInvocationUnit(new BasicInvocationUnit(referenceTracingValueFactory)),
                                  true,
                                  referenceTracingValueFactory),
             true);
    }


    /**
     * Creates a new ReferenceEscapeChecker.
     * @param partialEvaluator    the evaluator to be used for the analysis.
     * @param runPartialEvaluator specifies whether to run this evaluator on
     *                            every code attribute that is visited.
     */
    public ReferenceEscapeChecker(PartialEvaluator partialEvaluator,
                                  boolean          runPartialEvaluator)
    {
        this.partialEvaluator    = partialEvaluator;
        this.runPartialEvaluator = runPartialEvaluator;
    }


    /**
     * Returns whether the instance created or retrieved at the specified
     * instruction offset is escaping.
     */
    public boolean isInstanceEscaping(int instructionOffset)
    {
        return instanceEscaping[instructionOffset];
    }


    /**
     * Returns whether the instance created or retrieved at the specified
     * instruction offset is being returned.
     */
    public boolean isInstanceReturned(int instructionOffset)
    {
        return instanceReturned[instructionOffset];
    }


    /**
     * Returns whether the instance created or retrieved at the specified
     * instruction offset is being modified.
     */
    public boolean isInstanceModified(int instructionOffset)
    {
        return instanceModified[instructionOffset];
    }


    /**
     * Returns whether the instance created or retrieved at the specified
     * instruction offset is external to this method and its invoked methods.
     */
    public boolean isInstanceExternal(int instructionOffset)
    {
        return externalInstance[instructionOffset];
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Evaluate the method.
        if (runPartialEvaluator)
        {
            partialEvaluator.visitCodeAttribute(clazz, method, codeAttribute);
        }

        int codeLength = codeAttribute.u4codeLength;

        // Initialize the global arrays.
        instanceEscaping = ArrayUtil.ensureArraySize(instanceEscaping, codeLength, false);
        instanceReturned = ArrayUtil.ensureArraySize(instanceReturned, codeLength, false);
        instanceModified = ArrayUtil.ensureArraySize(instanceModified, codeLength, false);
        externalInstance = ArrayUtil.ensureArraySize(externalInstance, codeLength, false);

        // Mark the parameters and instances that are escaping from the code.
        codeAttribute.instructionsAccept(clazz, method, partialEvaluator.tracedInstructionFilter(this));

        if (DEBUG)
        {
            System.out.println();
            System.out.println("ReferenceEscapeChecker: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]");

            for (int index = 0; index < codeLength; index++)
            {
                if (partialEvaluator.isInstruction(index))
                {
                    System.out.println("  " +
                                       (instanceEscaping[index] ? 'E' : '.') +
                                       (instanceReturned[index] ? 'R' : '.') +
                                       (instanceModified[index] ? 'M' : '.') +
                                       (externalInstance[index] ? 'X' : '.') +
                                       ' ' +
                                       InstructionFactory.create(codeAttribute.code, index).toString(index));
                }
            }
        }
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        switch (simpleInstruction.opcode)
        {
            case InstructionConstants.OP_AASTORE:
                // Mark array reference values whose element is modified.
                markModifiedReferenceValues(offset,
                                            simpleInstruction.stackPopCount(clazz) - 1);

                // Mark reference values that are put in the array.
                markEscapingReferenceValues(offset, 0);
                break;

            case InstructionConstants.OP_IASTORE:
            case InstructionConstants.OP_LASTORE:
            case InstructionConstants.OP_FASTORE:
            case InstructionConstants.OP_DASTORE:
            case InstructionConstants.OP_BASTORE:
            case InstructionConstants.OP_CASTORE:
            case InstructionConstants.OP_SASTORE:
                // Mark array reference values whose element is modified.
                markModifiedReferenceValues(offset,
                                            simpleInstruction.stackPopCount(clazz) - 1);
                break;

            case InstructionConstants.OP_ARETURN:
                // Mark the returned reference values.
                markReturnedReferenceValues(offset, 0);
                break;

            case InstructionConstants.OP_ATHROW:
                // Mark the escaping reference values.
                markEscapingReferenceValues(offset, 0);
                break;
        }
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        switch (constantInstruction.opcode)
        {
            case InstructionConstants.OP_GETSTATIC:
            case InstructionConstants.OP_GETFIELD:
                // Mark external reference values.
                markExternalReferenceValue(offset);
                break;

            case InstructionConstants.OP_PUTSTATIC:
                // Mark reference values that are put in the field.
                markEscapingReferenceValues(offset, 0);
                break;

            case InstructionConstants.OP_PUTFIELD:
                // Mark reference reference values whose field is modified.
                markModifiedReferenceValues(offset,
                                            constantInstruction.stackPopCount(clazz) - 1);

                // Mark reference values that are put in the field.
                markEscapingReferenceValues(offset, 0);
                break;

            case InstructionConstants.OP_INVOKEVIRTUAL:
            case InstructionConstants.OP_INVOKESPECIAL:
            case InstructionConstants.OP_INVOKESTATIC:
            case InstructionConstants.OP_INVOKEINTERFACE:
                // Mark reference reference values that are modified as parameters
                // of the invoked method.
                // Mark reference values that are escaping as parameters
                // of the invoked method.
                // Mark escaped reference reference values in the invoked method.
                referencingMethod   = method;
                referencingOffset   = offset;
                referencingPopCount = constantInstruction.stackPopCount(clazz);
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                break;
        }
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        clazz.constantPoolEntryAccept(fieldrefConstant.u2classIndex, this);
    }


    public void visitAnyMethodrefConstant(Clazz clazz, RefConstant refConstant)
    {
        Method referencedMethod = (Method)refConstant.referencedMember;

        // Mark reference reference values that are passed to the method.
        for (int index = 0; index < referencingPopCount; index++)
        {
            int stackEntryIndex = referencingPopCount - index - 1;

            TracedStack stackBefore = partialEvaluator.getStackBefore(referencingOffset);
            Value       stackEntry  = stackBefore.getTop(stackEntryIndex);

            if (stackEntry.computationalType() == Value.TYPE_REFERENCE)
            {
                // Is the parameter escaping from the referenced method?
                if (referencedMethod == null ||
                    ParameterEscapeMarker.isParameterEscaping(referencedMethod, index))
                {
                    markEscapingReferenceValues(referencingOffset,
                                                stackEntryIndex);
                }

                // Is the parameter being modified in the referenced method?
                if (referencedMethod == null ||
                    ParameterEscapeMarker.isParameterModified(referencedMethod, index))
                {
                    markModifiedReferenceValues(referencingOffset,
                                                stackEntryIndex);
                }
            }
        }

        // Is the return value from the referenced method external?
        String returnType =
            ClassUtil.internalMethodReturnType(refConstant.getType(clazz));

        if (referencedMethod == null ||
            ((ClassUtil.isInternalClassType(returnType) ||
              ClassUtil.isInternalArrayType(returnType)) &&
             ParameterEscapeMarker.returnsExternalValues(referencedMethod)))
        {
            markExternalReferenceValue(referencingOffset);
        }
    }


    // Small utility methods.

    /**
     * Marks the producing offsets of the specified stack entry at the given
     * instruction offset.
     */
    private void markEscapingReferenceValues(int instructionOffset,
                                             int stackEntryIndex)
    {
        TracedStack stackBefore = partialEvaluator.getStackBefore(instructionOffset);
        Value       stackEntry  = stackBefore.getTop(stackEntryIndex);

        if (stackEntry.computationalType() == Value.TYPE_REFERENCE)
        {
            ReferenceValue referenceValue = stackEntry.referenceValue();

            // The null reference value may not have a trace value.
            if (referenceValue.isNull() != Value.ALWAYS)
            {
                markEscapingReferenceValues(referenceValue);
            }
        }
    }


    /**
     * Marks the producing offsets of the given traced reference value.
     */
    private void markEscapingReferenceValues(ReferenceValue referenceValue)
    {
        TracedReferenceValue   tracedReferenceValue   = (TracedReferenceValue)referenceValue;
        InstructionOffsetValue instructionOffsetValue = tracedReferenceValue.getTraceValue().instructionOffsetValue();

        int parameterCount = instructionOffsetValue.instructionOffsetCount();
        for (int index = 0; index < parameterCount; index++)
        {
            if (!instructionOffsetValue.isMethodParameter(index))
            {
                instanceEscaping[instructionOffsetValue.instructionOffset(index)] = true;
            }
        }
    }


    /**
     * Marks the producing offsets of the specified stack entry at the given
     * instruction offset.
     */
    private void markReturnedReferenceValues(int instructionOffset,
                                             int stackEntryIndex)
    {
        TracedStack    stackBefore    = partialEvaluator.getStackBefore(instructionOffset);
        ReferenceValue referenceValue = stackBefore.getTop(stackEntryIndex).referenceValue();

        // The null reference value may not have a trace value.
        if (referenceValue.isNull() != Value.ALWAYS)
        {
            markReturnedReferenceValues(referenceValue);
        }
    }


    /**
     * Marks the producing offsets of the given traced reference value.
     */
    private void markReturnedReferenceValues(ReferenceValue referenceValue)
    {
        TracedReferenceValue   tracedReferenceValue   = (TracedReferenceValue)referenceValue;
        InstructionOffsetValue instructionOffsetValue = tracedReferenceValue.getTraceValue().instructionOffsetValue();

        int parameterCount = instructionOffsetValue.instructionOffsetCount();
        for (int index = 0; index < parameterCount; index++)
        {
            if (!instructionOffsetValue.isMethodParameter(index))
            {
                instanceReturned[instructionOffsetValue.instructionOffset(index)] = true;
            }
        }
    }


    /**
     * Marks the producing offsets of the specified stack entry at the given
     * instruction offset.
     */
    private void markModifiedReferenceValues(int instructionOffset,
                                             int stackEntryIndex)
    {
        TracedStack    stackBefore    = partialEvaluator.getStackBefore(instructionOffset);
        ReferenceValue referenceValue = stackBefore.getTop(stackEntryIndex).referenceValue();

        // The null reference value may not have a trace value.
        if (referenceValue.isNull() != Value.ALWAYS)
        {
            markModifiedReferenceValues(referenceValue);
        }
    }


    /**
     * Marks the producing offsets of the given traced reference value.
     */
    private void markModifiedReferenceValues(ReferenceValue referenceValue)
    {
        TracedReferenceValue   tracedReferenceValue   = (TracedReferenceValue)referenceValue;
        InstructionOffsetValue instructionOffsetValue = tracedReferenceValue.getTraceValue().instructionOffsetValue();

        int parameterCount = instructionOffsetValue.instructionOffsetCount();
        for (int index = 0; index < parameterCount; index++)
        {
            if (!instructionOffsetValue.isMethodParameter(index))
            {
                instanceModified[instructionOffsetValue.instructionOffset(index)] = true;
            }
        }
    }


    /**
     * Marks the producing offsets of the specified stack entry at the given
     * instruction offset.
     */
    private void markExternalReferenceValue(int offset)
    {
        externalInstance[offset] = true;
    }
}