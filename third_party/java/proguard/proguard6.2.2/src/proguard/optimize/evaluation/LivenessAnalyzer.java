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
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.evaluation.BasicInvocationUnit;
import proguard.evaluation.value.*;
import proguard.util.ArrayUtil;

/**
 * This AttributeVisitor analyzes the liveness of the variables in the code
 * attributes that it visits, based on partial evaluation.
 *
 * @author Eric Lafortune
 */
public class LivenessAnalyzer
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor,
             ExceptionInfoVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("la") != null;
    //*/

    private static final int MAX_VARIABLES_SIZE = 64;

    private final PartialEvaluator     partialEvaluator;
    private final boolean              runPartialEvaluator;
    private final InitializationFinder initializationFinder;
    private final boolean              runInitializationFinder;

    private long[] isAliveBefore = new long[ClassConstants.TYPICAL_CODE_LENGTH];
    private long[] isAliveAfter  = new long[ClassConstants.TYPICAL_CODE_LENGTH];
    private long[] isCategory2   = new long[ClassConstants.TYPICAL_CODE_LENGTH];

    // Fields acting as global temporary variables.
    private boolean checkAgain;
    private long    alive;


    /**
     * Creates a new LivenessAnalyzer.
     */
    public LivenessAnalyzer()
    {
        this(new ReferenceTracingValueFactory(new BasicValueFactory()));
    }


    /**
     * Creates a new LivenessAnalyzer. This private constructor gets around
     * the constraint that it's not allowed to add statements before calling
     * 'this'.
     */
    private LivenessAnalyzer(ReferenceTracingValueFactory referenceTracingValueFactory)
    {
        this(new PartialEvaluator(referenceTracingValueFactory,
                                  new ReferenceTracingInvocationUnit(new BasicInvocationUnit(referenceTracingValueFactory)),
                                  true,
                                  referenceTracingValueFactory),
             true);
    }


    /**
     * Creates a new LivenessAnalyzer. This private constructor gets around
     * the constraint that it's not allowed to add statements before calling
     * 'this'.
     */
    private LivenessAnalyzer(PartialEvaluator partialEvaluator,
                             boolean          runPartialEvaluator)
    {
        this(partialEvaluator,
             runPartialEvaluator,
             new InitializationFinder(partialEvaluator, false),
             true);
    }


    /**
     * Creates a new LivenessAnalyzer that will use the given partial evaluator
     * and initialization finder.
     * @param partialEvaluator        the evaluator to be used for the analysis.
     * @param runPartialEvaluator     specifies whether to run this evaluator on
     *                                every code attribute that is visited.
     * @param initializationFinder    the initialization finder to be used for
     *                                the analysis.
     * @param runInitializationFinder specifies whether to run this
     *                                initialization finder on every code
     *                                attribute that is visited.
     */
    public LivenessAnalyzer(PartialEvaluator     partialEvaluator,
                            boolean              runPartialEvaluator,
                            InitializationFinder initializationFinder,
                            boolean              runInitializationFinder)
    {
        this.partialEvaluator        = partialEvaluator;
        this.runPartialEvaluator     = runPartialEvaluator;
        this.initializationFinder    = initializationFinder;
        this.runInitializationFinder = runInitializationFinder;
    }


    /**
     * Returns whether the instruction at the given offset has ever been
     * executed during the partial evaluation.
     */
    public boolean isTraced(int instructionOffset)
    {
        return partialEvaluator.isTraced(instructionOffset);
    }


    /**
     * Returns whether the specified variable is alive before the instruction
     * at the given offset.
     */
    public boolean isAliveBefore(int instructionOffset, int variableIndex)
    {
        return variableIndex >= MAX_VARIABLES_SIZE ?
            partialEvaluator.getVariablesBefore(instructionOffset).getValue(variableIndex) != null :
            (isAliveBefore[instructionOffset] & (1L << variableIndex)) != 0;
    }


    /**
     * Sets whether the specified variable is alive before the instruction
     * at the given offset.
     */
    public void setAliveBefore(int instructionOffset, int variableIndex, boolean alive)
    {
        if (variableIndex < MAX_VARIABLES_SIZE)
        {
            if (alive)
            {
                isAliveBefore[instructionOffset] |= 1L << variableIndex;
            }
            else
            {
                isAliveBefore[instructionOffset] &= ~(1L << variableIndex);
            }
        }
    }


    /**
     * Returns whether the specified variable is alive after the instruction
     * at the given offset.
     */
    public boolean isAliveAfter(int instructionOffset, int variableIndex)
    {
        return variableIndex >= MAX_VARIABLES_SIZE ?
            partialEvaluator.getVariablesAfter(instructionOffset).getValue(variableIndex) != null :
            (isAliveAfter[instructionOffset] & (1L << variableIndex)) != 0;
    }


    /**
     * Sets whether the specified variable is alive after the instruction
     * at the given offset.
     */
    public void setAliveAfter(int instructionOffset, int variableIndex, boolean alive)
    {
        if (variableIndex < MAX_VARIABLES_SIZE)
        {
            if (alive)
            {
                isAliveAfter[instructionOffset] |= 1L << variableIndex;
            }
            else
            {
                isAliveAfter[instructionOffset] &= ~(1L << variableIndex);
            }
        }
    }


    /**
     * Returns whether the specified variable takes up two entries after the
     * instruction at the given offset.
     */
    public boolean isCategory2(int instructionOffset, int variableIndex)
    {
        return variableIndex >= MAX_VARIABLES_SIZE ?
            partialEvaluator.getVariablesBefore(instructionOffset).getValue(variableIndex) != null &&
            partialEvaluator.getVariablesBefore(instructionOffset).getValue(variableIndex).isCategory2() :
            (isCategory2[instructionOffset] & (1L << variableIndex)) != 0;
    }


    /**
     * Sets whether the specified variable takes up two entries after the
     * instruction at the given offset.
     */
    public void setCategory2(int instructionOffset, int variableIndex, boolean category2)
    {
        if (variableIndex < MAX_VARIABLES_SIZE)
        {
            if (category2)
            {
                isCategory2[instructionOffset] |= 1L << variableIndex;
            }
            else
            {
                isCategory2[instructionOffset] &= ~(1L << variableIndex);
            }
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
//        DEBUG =
//            clazz.getName().equals("abc/Def") &&
//            method.getName(clazz).equals("abc");

        if (DEBUG)
        {
            System.out.println();
            System.out.println("Liveness analysis: "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz));
        }

        int codeLength    = codeAttribute.u4codeLength;
        int variablesSize = codeAttribute.u2maxLocals;

        // Initialize the global arrays.
        isAliveBefore = ArrayUtil.ensureArraySize(isAliveBefore, codeLength, 0L);
        isAliveAfter  = ArrayUtil.ensureArraySize(isAliveAfter,  codeLength, 0L);
        isCategory2   = ArrayUtil.ensureArraySize(isCategory2,   codeLength, 0L);

        // Evaluate the method.
        if (runPartialEvaluator)
        {
            partialEvaluator.visitCodeAttribute(clazz, method, codeAttribute);
        }

        if (runInitializationFinder)
        {
            initializationFinder.visitCodeAttribute(clazz, method, codeAttribute);
        }

        // We'll only really analyze the first 64 variables.
        if (variablesSize > MAX_VARIABLES_SIZE)
        {
            variablesSize = MAX_VARIABLES_SIZE;
        }

        // Mark liveness blocks, as many times as necessary.
        do
        {
            checkAgain = false;
            alive      = 0L;

            // Loop over all traced instructions, backward.
            for (int offset = codeLength - 1; offset >= 0; offset--)
            {
                if (partialEvaluator.isTraced(offset))
                {
                    // Update the liveness based on the branch targets.
                    InstructionOffsetValue branchTargets = partialEvaluator.branchTargets(offset);
                    if (branchTargets != null)
                    {
                        // Update the liveness right after the branch instruction.
                        alive = combinedLiveness(branchTargets);
                    }

                    // Merge the current liveness.
                    alive |= isAliveAfter[offset];

                    // Update the liveness after the instruction.
                    isAliveAfter[offset] = alive;

                    // Update the current liveness based on the instruction.
                    codeAttribute.instructionAccept(clazz, method, offset, this);

                    // Merge the current liveness.
                    alive |= isAliveBefore[offset];

                    // Update the liveness before the instruction.
                    if ((~isAliveBefore[offset] & alive) != 0L)
                    {
                        isAliveBefore[offset] = alive;

                        // Do we have to check again after this loop?
                        InstructionOffsetValue branchOrigins = partialEvaluator.branchOrigins(offset);
                        if (branchOrigins != null &&
                            offset < branchOrigins.maximumValue())
                        {
                            checkAgain = true;
                        }
                    }
                }
            }

            // Account for the liveness at the start of the exception handlers.
            codeAttribute.exceptionsAccept(clazz, method, this);
        }
        while (checkAgain);

        // Loop over all instructions, to mark variables that take up two entries.
        for (int offset = 0; offset < codeLength; offset++)
        {
            if (partialEvaluator.isTraced(offset))
            {
                // Loop over all variables.
                for (int variableIndex = 0; variableIndex < variablesSize; variableIndex++)
                {
                    // Is the variable alive and a category 2 type?
                    if (isAliveBefore(offset, variableIndex))
                    {
                        Value value = partialEvaluator.getVariablesBefore(offset).getValue(variableIndex);
                        if (value != null && value.isCategory2())
                        {
                            // Mark it as such.
                            setCategory2(offset, variableIndex, true);

                            // Mark the next variable as well.
                            setAliveBefore(offset, variableIndex + 1, true);
                            setCategory2(  offset, variableIndex + 1, true);
                        }
                    }

                    // Is the variable alive and a category 2 type?
                    if (isAliveAfter(offset, variableIndex))
                    {
                        Value value = partialEvaluator.getVariablesAfter(offset).getValue(variableIndex);
                        if (value != null && value.isCategory2())
                        {
                            // Mark it as such.
                            setCategory2(offset, variableIndex, true);

                            // Mark the next variable as well.
                            setAliveAfter(offset, variableIndex + 1, true);
                            setCategory2( offset, variableIndex + 1, true);
                        }
                    }
                }
            }
        }

        if (DEBUG)
        {
            // Loop over all instructions.
            for (int offset = 0; offset < codeLength; offset++)
            {
                if (partialEvaluator.isTraced(offset))
                {
                    long aliveBefore = isAliveBefore[offset];
                    long aliveAfter  = isAliveAfter[offset];
                    long category2   = isCategory2[offset];

                    // Print out the liveness of all variables before the instruction.
                    for (int variableIndex = 0; variableIndex < variablesSize; variableIndex++)
                    {
                        long variableMask = (1L << variableIndex);
                        System.out.print((aliveBefore & variableMask) == 0L ? '.' :
                                         (category2   & variableMask) == 0L ? 'x' :
                                                                              '*');
                    }

                    // Print out the instruction itself.
                    System.out.println(" "+ InstructionFactory.create(codeAttribute.code, offset).toString(offset));

                    // Print out the liveness of all variables after the instruction.
                    for (int variableIndex = 0; variableIndex < variablesSize; variableIndex++)
                    {
                        long variableMask = (1L << variableIndex);
                        System.out.print((aliveAfter & variableMask) == 0L ? '.' :
                                         (category2  & variableMask) == 0L ? 'x' :
                                                                             '=');
                    }

                    System.out.println();
                }
            }
        }
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        int variableIndex = variableInstruction.variableIndex;
        if (variableIndex < MAX_VARIABLES_SIZE)
        {
            long livenessMask = 1L << variableIndex;

            // Is it a load instruction or a store instruction?
            if (variableInstruction.isLoad())
            {
                // Start marking the variable before the load instruction.
                alive |= livenessMask;
            }
            else
            {
                // Stop marking the variable before the store instruction.
                alive &= ~livenessMask;

                // But do mark the variable right after the store instruction.
                isAliveAfter[offset] |= livenessMask;
            }
        }
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        // Special case: variable 0 ('this') in an initializer has to be alive
        // as long as it hasn't been initialized.
        if (offset == initializationFinder.superInitializationOffset())
        {
            alive |= 1L;
        }
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        // Are any variables alive at the start of the handler?
        long alive = isAliveBefore[exceptionInfo.u2handlerPC];
        if (alive != 0L)
        {
            // Set the same liveness flags for the entire try block.
            int startOffset = exceptionInfo.u2startPC;
            int endOffset   = exceptionInfo.u2endPC;

            for (int offset = startOffset; offset < endOffset; offset++)
            {
                if (partialEvaluator.isTraced(offset))
                {
                    if ((~(isAliveBefore[offset] & isAliveAfter[offset]) & alive) != 0L)
                    {
                        isAliveBefore[offset] |= alive;
                        isAliveAfter[offset]  |= alive;

                        // Check again after having marked this try block.
                        checkAgain = true;
                    }
                }
            }
        }
    }


    // Small utility methods.

    /**
     * Returns the combined liveness mask of the variables right before the
     * specified instruction offsets.
     */
    private long combinedLiveness(InstructionOffsetValue instructionOffsetValue)
    {
        long alive = 0L;

        int count = instructionOffsetValue.instructionOffsetCount();
        for (int index = 0; index < count; index++)
        {
            alive |= isAliveBefore[instructionOffsetValue.instructionOffset(index)];
        }

        return alive;
    }
}
