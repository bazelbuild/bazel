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
package proguard.optimize.evaluation;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.instruction.InstructionFactory;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.evaluation.BasicInvocationUnit;
import proguard.evaluation.value.*;
import proguard.util.ArrayUtil;

/**
 * This AttributeVisitor links 'new' instructions and their corresponding
 * initializers in the CodeAttribute objects that it visits.
 *
 * @author Eric Lafortune
 */
public class InitializationFinder
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("if") != null;
    //*/

    public static final int NONE = -1;

    private final PartialEvaluator partialEvaluator;
    private final boolean          runPartialEvaluator;

    private int                      superInitializationOffset;
    private int[]                    initializationOffsets = new int[ClassConstants.TYPICAL_CODE_LENGTH];
    private InstructionOffsetValue[] uninitializedOffsets  = new InstructionOffsetValue[ClassConstants.TYPICAL_CODE_LENGTH];


    /**
     * Creates a new InitializationFinder.
     */
    public InitializationFinder()
    {
        this(new ReferenceTracingValueFactory(new BasicValueFactory()));
    }


    /**
     * Creates a new InitializationFinder. This private constructor gets around
     * the constraint that it's not allowed to add statements before calling
     * 'this'.
     */
    private InitializationFinder(ReferenceTracingValueFactory referenceTracingValueFactory)
    {
        this(new PartialEvaluator(referenceTracingValueFactory,
                                  new ReferenceTracingInvocationUnit(new BasicInvocationUnit(referenceTracingValueFactory)),
                                  true,
                                  referenceTracingValueFactory),
             true);
    }


    /**
     * Creates a new InitializationFinder that will use the given partial
     * evaluator.
     * @param partialEvaluator    the evaluator to be used for the analysis.
     * @param runPartialEvaluator specifies whether to run this evaluator on
     *                            every code attribute that is visited.
     */
    public InitializationFinder(PartialEvaluator partialEvaluator,
                                boolean          runPartialEvaluator)
    {
        this.partialEvaluator    = partialEvaluator;
        this.runPartialEvaluator = runPartialEvaluator;
    }


    /**
     * Returns whether the method is an instance initializer, in the
     * CodeAttribute that was visited most recently.
     */
    public boolean isInitializer()
    {
        return superInitializationOffset != NONE;
    }


    /**
     * Returns the instruction offset at which this initializer is calling
     * the "super" or "this" initializer method, or <code>NONE</code> if it is
     * not an initializer.
     */
    public int superInitializationOffset()
    {
        return superInitializationOffset;
    }


//    /**
//     * Returns whether the instruction at the given offset is a 'new'
//     * instruction.
//     */
//    public boolean isNew(int offset)
//    {
//        return initializationOffsets[offset] != NONE;
//    }
//
//
//    /**
//     * Returns the instruction offset at which the object instance that is
//     * created at the given 'new' instruction offset is initialized, or
//     * <code>NONE</code> if it is not being created.
//     */
//    public int initializationOffset(int creationOffset)
//    {
//        return initializationOffsets[creationOffset];
//    }


    /**
     * Returns the 'new' instruction offset at which the object instance is
     * created that is initialized at the given offset.
     */
    public int creationOffset(int initializationOffset)
    {
        return creationOffsetValue(initializationOffset).instructionOffset(0);
    }


    /**
     * Returns whether the specified stack entry is initialized.
     */
    public boolean isInitializedBefore(int offset, int stackEntryIndexBottom)
    {
        InstructionOffsetValue creationOffsetValue =
            creationOffsetValue(offset, stackEntryIndexBottom);

        return isInitializedBefore(offset, creationOffsetValue);
    }


    /**
     * Returns whether the given creation offset is initialized before the given
     * offset.
     */
    public boolean isInitializedBefore(int                    offset,
                                       InstructionOffsetValue creationOffsetValue)
    {
        return !uninitializedOffsets[offset].contains(creationOffsetValue.instructionOffset(0));
    }


    /**
     * Returns whether the instruction at the given offset is the special
     * invocation of an instance initializer.
     */
    public boolean isInitializer(int offset)
    {
        return partialEvaluator.isInitializer(offset);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
//        DEBUG =
//            clazz.getName().equals("abc/Def") &&
//            method.getName(clazz).equals("abc");

        int codeLength = codeAttribute.u4codeLength;

        superInitializationOffset = NONE;

        // Make sure the global arrays are sufficiently large.
        initializationOffsets = ArrayUtil.ensureArraySize(initializationOffsets, codeLength, NONE);
        uninitializedOffsets  = ArrayUtil.ensureArraySize(uninitializedOffsets,codeLength, null);

        // Evaluate the method.
        if (runPartialEvaluator)
        {
            partialEvaluator.visitCodeAttribute(clazz, method, codeAttribute);
        }

        // Loop over all instructions. This is sufficient, because the JVM
        // specifications don't allow uninitialized instances on the stack or
        // in variables when branching backward. JVMs without preverification
        // and the Dalvik VM do allow it in practice.
        InstructionOffsetValue currentUninitializedOffsets = method.getName(clazz).equals(ClassConstants.METHOD_NAME_INIT) ?
            new InstructionOffsetValue(InstructionOffsetValue.METHOD_PARAMETER) :
            InstructionOffsetValue.EMPTY_VALUE;

        for (int offset = 0; offset < codeLength; offset++)
        {
            if (partialEvaluator.isTraced(offset))
            {
                // Exception handlers start without uninitialized instances
                // (on the stack or in variables).
                if (partialEvaluator.isExceptionHandler(offset))
                {
                    currentUninitializedOffsets = InstructionOffsetValue.EMPTY_VALUE;
                }

                // Check if the uninitialized creation offsets have been set
                // before (because of a forward branch).
                if (uninitializedOffsets[offset] != null)
                {
                    // Continue using them.
                    currentUninitializedOffsets = uninitializedOffsets[offset];
                }
                else
                {
                    uninitializedOffsets[offset] = currentUninitializedOffsets;
                }

                // Is it a 'new' instruction?
                if (partialEvaluator.isCreation(offset))
                {
                    // Add its offset to the current list.
                    currentUninitializedOffsets =
                        currentUninitializedOffsets.add(offset);
                }
                // Is it an instance initialization?
                else if (partialEvaluator.isInitializer(offset))
                {
                    // Remove its creation offset from the current list.
                    InstructionOffsetValue creationOffsetValue =
                        creationOffsetValue(offset);

                    int creationOffset =
                        creationOffsetValue.instructionOffset(0);

                    if (creationOffsetValue.isMethodParameter(0))
                    {
                        // Remember the super initialization offset of the
                        // initializer method.
                        superInitializationOffset = offset;
                    }
                    else
                    {
                        // Remember the instance initialization for the 'new'
                        // instruction.
                        initializationOffsets[creationOffset] = offset;
                    }

                    currentUninitializedOffsets =
                        currentUninitializedOffsets.remove(creationOffset);
                }

                // Propagate the uninitialized creation offsets to the forward
                // branch targets, if any.
                InstructionOffsetValue branchTargets =
                    partialEvaluator.branchTargets(offset);

                if (branchTargets != null)
                {
                    for (int branchIndex = 0; branchIndex < branchTargets.instructionOffsetCount(); branchIndex++)
                    {
                        int branchOffset = branchTargets.instructionOffset(branchIndex);
                        if (branchOffset > offset)
                        {
                            uninitializedOffsets[branchOffset] = currentUninitializedOffsets;
                        }
                    }

                    currentUninitializedOffsets = InstructionOffsetValue.EMPTY_VALUE;
                }
            }
        }

        if (DEBUG)
        {
            System.out.println();
            System.out.println("InitializationFinder: "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz));

            for (int offset = 0; offset < codeLength; offset++)
            {
                if (partialEvaluator.isInstruction(offset))
                {
                    System.out.println((initializationOffsets[offset] >= 0 ? "i"+initializationOffsets[offset] : "   ") +
                                       (uninitializedOffsets[offset] != null &&
                                        uninitializedOffsets[offset].instructionOffsetCount() > 0 ? " u"+uninitializedOffsets[offset] : "    ") +
                                       (isInitializer(offset) ? " "+creationOffsetValue(offset) : "    ") + " " +
                                       InstructionFactory.create(codeAttribute.code, offset).toString(offset));
                }
            }
        }
    }


    // Small utility methods.

    /**
     * Returns the 'new' instruction offset value (or method parameter) at
     * which the object instance is created that is initialized at the given
     * offset.
     */
    private InstructionOffsetValue creationOffsetValue(int initializationOffset)
    {
        int stackEntryIndexBottom =
            partialEvaluator.getStackAfter(initializationOffset).size();

        return creationOffsetValue(initializationOffset, stackEntryIndexBottom);
    }


    /**
     * Returns the 'new' instruction offset value (or method parameter) of
     * the specified stack entry.
     */
    private InstructionOffsetValue creationOffsetValue(int instructionOffset,
                                                       int stackEntryIndexBottom)
    {
        // Get the reference value of the new instance.
        ReferenceValue newReferenceValue =
            partialEvaluator.getStackBefore(instructionOffset).getBottom(stackEntryIndexBottom).referenceValue();

        // It's a traced reference.
        TracedReferenceValue tracedReferenceValue =
            (TracedReferenceValue)newReferenceValue;

        // Get the trace value.
        return tracedReferenceValue.getTraceValue().instructionOffsetValue();
    }
}
