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
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.evaluation.*;
import proguard.evaluation.value.*;
import proguard.optimize.peephole.BranchTargetFinder;

import java.util.Arrays;

/**
 * This AttributeVisitor performs partial evaluation on the code attributes
 * that it visits.
 *
 * @author Eric Lafortune
 */
public class PartialEvaluator
extends      SimplifiedVisitor
implements   AttributeVisitor,
             ExceptionInfoVisitor
{
    //*
    private static final boolean DEBUG         = false;
    private static final boolean DEBUG_RESULTS = false;
    /*/
    public static boolean DEBUG         = System.getProperty("pe") != null;
    public static boolean DEBUG_RESULTS = DEBUG;
    //*/

    private static final int MAXIMUM_EVALUATION_COUNT = 5;

    public static final int NONE            = -2;
    public static final int AT_METHOD_ENTRY = -1;
    public static final int AT_CATCH_ENTRY  = -1;

    private final ValueFactory       valueFactory;
    private final InvocationUnit     invocationUnit;
    private final boolean            evaluateAllCode;
    private final InstructionVisitor extraInstructionVisitor;

    private InstructionOffsetValue[] branchOriginValues  = new InstructionOffsetValue[ClassConstants.TYPICAL_CODE_LENGTH];
    private InstructionOffsetValue[] branchTargetValues  = new InstructionOffsetValue[ClassConstants.TYPICAL_CODE_LENGTH];
    private TracedVariables[]        variablesBefore     = new TracedVariables[ClassConstants.TYPICAL_CODE_LENGTH];
    private TracedStack[]            stacksBefore        = new TracedStack[ClassConstants.TYPICAL_CODE_LENGTH];
    private TracedVariables[]        variablesAfter      = new TracedVariables[ClassConstants.TYPICAL_CODE_LENGTH];
    private TracedStack[]            stacksAfter         = new TracedStack[ClassConstants.TYPICAL_CODE_LENGTH];
    private boolean[]                generalizedContexts = new boolean[ClassConstants.TYPICAL_CODE_LENGTH];
    private int[]                    evaluationCounts    = new int[ClassConstants.TYPICAL_CODE_LENGTH];
    private boolean                  evaluateExceptions;
    private int                      codeLength;

    private final BasicBranchUnit    branchUnit;
    private final BranchTargetFinder branchTargetFinder;

    private final java.util.Stack callingInstructionBlockStack;
    private final java.util.Stack instructionBlockStack = new java.util.Stack();


    /**
     * Creates a simple PartialEvaluator.
     */
    public PartialEvaluator()
    {
        this(new BasicValueFactory());
    }


    /**
     * Creates a new PartialEvaluator.
     * @param valueFactory    the value factory that will create all values
     *                        during evaluation.
     */
    public PartialEvaluator(ValueFactory valueFactory)
    {
        this(valueFactory,
             new BasicInvocationUnit(valueFactory),
             true);
    }


    /**
     * Creates a new PartialEvaluator.
     * @param valueFactory    the value factory that will create all values
     *                        during the evaluation.
     * @param invocationUnit  the invocation unit that will handle all
     *                        communication with other fields and methods.
     * @param evaluateAllCode a flag that specifies whether all casts, branch
     *                        targets, and exception handlers should be
     *                        evaluated, even if they are unnecessary or
     *                        unreachable.
     */
    public PartialEvaluator(ValueFactory   valueFactory,
                            InvocationUnit invocationUnit,
                            boolean        evaluateAllCode)
    {
        this(valueFactory,
             invocationUnit,
             evaluateAllCode,
             null);
    }


    /**
     * Creates a new PartialEvaluator.
     * @param valueFactory            the value factory that will create all
     *                                values during the evaluation.
     * @param invocationUnit          the invocation unit that will handle all
     *                                communication with other fields and
     *                                methods.
     * @param evaluateAllCode         a flag that specifies whether all branch
     *                                targets and exception handlers should be
     *                                evaluated, even if they are unreachable.
     * @param extraInstructionVisitor an optional extra visitor for all
     *                                instructions right before they are
     *                                executed.
     */
    public PartialEvaluator(ValueFactory       valueFactory,
                            InvocationUnit     invocationUnit,
                            boolean            evaluateAllCode,
                            InstructionVisitor extraInstructionVisitor)
    {
        this(valueFactory,
             invocationUnit,
             evaluateAllCode,
             extraInstructionVisitor,
             evaluateAllCode ?
                 new BasicBranchUnit() :
                 new TracedBranchUnit(),
             new BranchTargetFinder(),
             null);
    }


    /**
     * Creates a new PartialEvaluator, based on an existing one.
     * @param partialEvaluator the subroutine calling partial evaluator.
     */
    private PartialEvaluator(PartialEvaluator partialEvaluator)
    {
        this(partialEvaluator.valueFactory,
             partialEvaluator.invocationUnit,
             partialEvaluator.evaluateAllCode,
             partialEvaluator.extraInstructionVisitor,
             partialEvaluator.branchUnit,
             partialEvaluator.branchTargetFinder,
             partialEvaluator.instructionBlockStack);
    }


    /**
     * Creates a new PartialEvaluator.
     * @param valueFactory                 the value factory that will create
     *                                     all values during evaluation.
     * @param invocationUnit               the invocation unit that will handle
     *                                     all communication with other fields
     *                                     and methods.
     * @param evaluateAllCode              a flag that specifies whether all
     *                                     casts, branch targets, and exception
     *                                     handlers should be evaluated, even
     *                                     if they are unnecessary or
     *                                     unreachable.
     * @param branchUnit                   the branch unit that will handle all
     *                                     branches.
     * @param branchTargetFinder           the utility class that will find all
     *                                     branches.
     * @param callingInstructionBlockStack the stack of instruction blocks to
     *                                     be evaluated
     */
    private PartialEvaluator(ValueFactory       valueFactory,
                             InvocationUnit     invocationUnit,
                             boolean            evaluateAllCode,
                             InstructionVisitor extraInstructionVisitor,
                             BasicBranchUnit    branchUnit,
                             BranchTargetFinder branchTargetFinder,
                             java.util.Stack    callingInstructionBlockStack)
    {
        this.valueFactory                 = valueFactory;
        this.invocationUnit               = invocationUnit;
        this.evaluateAllCode              = evaluateAllCode;
        this.extraInstructionVisitor      = extraInstructionVisitor;
        this.branchUnit                   = branchUnit;
        this.branchTargetFinder           = branchTargetFinder;
        this.callingInstructionBlockStack = callingInstructionBlockStack == null ?
            this.instructionBlockStack :
            callingInstructionBlockStack;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
//        DEBUG = DEBUG_RESULTS =
//            clazz.getName().equals("abc/Def") &&
//            method.getName(clazz).equals("abc");

        // TODO: Remove this when the partial evaluator has stabilized.
        // Catch any unexpected exceptions from the actual visiting method.
        try
        {
            // Process the code.
            visitCodeAttribute0(clazz, method, codeAttribute);
        }
        catch (RuntimeException ex)
        {
            System.err.println("Unexpected error while performing partial evaluation:");
            System.err.println("  Class       = ["+clazz.getName()+"]");
            System.err.println("  Method      = ["+method.getName(clazz)+method.getDescriptor(clazz)+"]");
            System.err.println("  Exception   = ["+ex.getClass().getName()+"] ("+ex.getMessage()+")");

            if (DEBUG)
            {
                method.accept(clazz, new ClassPrinter());

                System.out.println("Evaluation results:");

                int offset = 0;
                do
                {
                    if (isBranchOrExceptionTarget(offset))
                    {
                        System.out.println("Branch target from ["+branchOriginValues[offset]+"]:");
                        if (isTraced(offset))
                        {
                            System.out.println("  Vars:  "+variablesBefore[offset]);
                            System.out.println("  Stack: "+stacksBefore[offset]);
                        }
                    }

                    Instruction instruction = InstructionFactory.create(codeAttribute.code,
                                                                        offset);
                    System.out.println(instruction.toString(offset));

                    if (isTraced(offset))
                    {
//                        int initializationOffset = branchTargetFinder.initializationOffset(offset);
//                        if (initializationOffset != NONE)
//                        {
//                            System.out.println("     is to be initialized at ["+initializationOffset+"]");
//                        }

                        InstructionOffsetValue branchTargets = branchTargets(offset);
                        if (branchTargets != null)
                        {
                            System.out.println("     has overall been branching to "+branchTargets);
                        }

                        System.out.println("  Vars:  "+variablesAfter[offset]);
                        System.out.println("  Stack: "+stacksAfter[offset]);
                    }

                    offset += instruction.length(offset);
                }
                while (offset < codeAttribute.u4codeLength);
            }

            throw ex;
        }
    }


    public void visitCodeAttribute0(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Evaluate the instructions, starting at the entry point.
        if (DEBUG)
        {
            System.out.println();
            System.out.println("Partial evaluation: "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz));
            System.out.println("  Max locals = "+codeAttribute.u2maxLocals);
            System.out.println("  Max stack  = "+codeAttribute.u2maxStack);
        }

        // Reuse the existing variables and stack objects, ensuring the right size.
        TracedVariables variables = new TracedVariables(codeAttribute.u2maxLocals);
        TracedStack     stack     = new TracedStack(codeAttribute.u2maxStack);

        // Initialize the reusable arrays and variables.
        initializeArrays(codeAttribute);
        initializeParameters(clazz, method, codeAttribute, variables);

        // Reset stacks.
        instructionBlockStack.clear();
        callingInstructionBlockStack.clear();

        // Find all instruction offsets,...
        codeAttribute.accept(clazz, method, branchTargetFinder);

        // Start executing the first instruction block.
        evaluateInstructionBlockAndExceptionHandlers(clazz,
                                                     method,
                                                     codeAttribute,
                                                     variables,
                                                     stack,
                                                     0,
                                                     codeAttribute.u4codeLength);

        if (DEBUG_RESULTS)
        {
            System.out.println("Evaluation results:");

            int offset = 0;
            do
            {
                if (isBranchOrExceptionTarget(offset))
                {
                    System.out.println("Branch target from ["+branchOriginValues[offset]+"]:");
                    if (isTraced(offset))
                    {
                        System.out.println("  Vars:  "+variablesBefore[offset]);
                        System.out.println("  Stack: "+stacksBefore[offset]);
                    }
                }

                Instruction instruction = InstructionFactory.create(codeAttribute.code,
                                                                    offset);
                System.out.println(instruction.toString(offset));

                if (isTraced(offset))
                {
//                    int initializationOffset = branchTargetFinder.initializationOffset(offset);
//                    if (initializationOffset != NONE)
//                    {
//                        System.out.println("     is to be initialized at ["+initializationOffset+"]");
//                    }

                    InstructionOffsetValue branchTargets = branchTargets(offset);
                    if (branchTargets != null)
                    {
                        System.out.println("     has overall been branching to "+branchTargets);
                    }

                    System.out.println("  Vars:  "+variablesAfter[offset]);
                    System.out.println("  Stack: "+stacksAfter[offset]);
                }

                offset += instruction.length(offset);
            }
            while (offset < codeAttribute.u4codeLength);
        }
    }


    /**
     * Returns whether a block of instructions is ever used.
     */
    public boolean isTraced(int startOffset, int endOffset)
    {
        for (int index = startOffset; index < endOffset; index++)
        {
            if (isTraced(index))
            {
                return true;
            }
        }

        return false;
    }


    /**
     * Returns whether the instruction at the given offset has ever been
     * executed during the partial evaluation.
     */
    public boolean isTraced(int instructionOffset)
    {
        return evaluationCounts[instructionOffset] > 0;
    }


    /**
     * Returns whether there is an instruction at the given offset.
     */
    public boolean isInstruction(int instructionOffset)
    {
        return branchTargetFinder.isInstruction(instructionOffset);
    }


    /**
     * Returns whether the instruction at the given offset is the target of
     * any kind.
     */
    public boolean isTarget(int instructionOffset)
    {
        return branchTargetFinder.isTarget(instructionOffset);
    }


    /**
     * Returns whether the instruction at the given offset is the origin of a
     * branch instruction.
     */
    public boolean isBranchOrigin(int instructionOffset)
    {
        return branchTargetFinder.isBranchOrigin(instructionOffset);
    }


    /**
     * Returns whether the instruction at the given offset is the target of a
     * branch instruction.
     */
    public boolean isBranchTarget(int instructionOffset)
    {
        return branchTargetFinder.isBranchTarget(instructionOffset);
    }


    /**
     * Returns whether the instruction at the given offset is the target of a
     * branch instruction or an exception.
     */
    public boolean isBranchOrExceptionTarget(int instructionOffset)
    {
        return branchTargetFinder.isBranchTarget(instructionOffset) ||
               branchTargetFinder.isExceptionHandler(instructionOffset);
    }


    /**
     * Returns whether the instruction at the given offset is the start of
     * an exception handler.
     */
    public boolean isExceptionHandler(int instructionOffset)
    {
        return branchTargetFinder.isExceptionHandler(instructionOffset);
    }


    /**
     * Returns whether the instruction at the given offset is the start of a
     * subroutine.
     */
    public boolean isSubroutineStart(int instructionOffset)
    {
        return branchTargetFinder.isSubroutineStart(instructionOffset);
    }


    /**
     * Returns whether the instruction at the given offset is a subroutine
     * invocation.
     */
    public boolean isSubroutineInvocation(int instructionOffset)
    {
        return branchTargetFinder.isSubroutineInvocation(instructionOffset);
    }


    /**
     * Returns whether the instruction at the given offset is part of a
     * subroutine.
     */
    public boolean isSubroutine(int instructionOffset)
    {
        return branchTargetFinder.isSubroutine(instructionOffset);
    }


    /**
     * Returns whether the subroutine at the given offset is ever returning
     * by means of a regular 'ret' instruction.
     */
    public boolean isSubroutineReturning(int instructionOffset)
    {
        return branchTargetFinder.isSubroutineReturning(instructionOffset);
    }


    /**
     * Returns the offset after the subroutine that starts at the given
     * offset.
     */
    public int subroutineEnd(int instructionOffset)
    {
        return branchTargetFinder.subroutineEnd(instructionOffset);
    }


//    /**
//     * Returns the instruction offset at which the object instance that is
//     * created at the given 'new' instruction offset is initialized, or
//     * <code>NONE</code> if it is not being created.
//     */
//    public int initializationOffset(int instructionOffset)
//    {
//        return branchTargetFinder.initializationOffset(instructionOffset);
//    }


//    /**
//     * Returns whether the method is an instance initializer.
//     */
//    public boolean isInitializer()
//    {
//        return branchTargetFinder.isInitializer();
//    }


//    /**
//     * Returns the instruction offset at which this initializer is calling
//     * the "super" or "this" initializer method, or <code>NONE</code> if it is
//     * not an initializer.
//     */
//    public int superInitializationOffset()
//    {
//        return branchTargetFinder.superInitializationOffset();
//    }


    /**
     * Returns whether the instruction at the given offset creates a new,
     * uninitialized instance.
     */
    public boolean isCreation(int offset)
    {
        return branchTargetFinder.isCreation(offset);
    }


    /**
     * Returns whether the instruction at the given offset is the special
     * invocation of an instance initializer.
     */
    public boolean isInitializer(int offset)
    {
        return branchTargetFinder.isInitializer(offset);
    }


    /**
     * Returns the variables before execution of the instruction at the given
     * offset.
     */
    public TracedVariables getVariablesBefore(int instructionOffset)
    {
        return variablesBefore[instructionOffset];
    }


    /**
     * Returns the variables after execution of the instruction at the given
     * offset.
     */
    public TracedVariables getVariablesAfter(int instructionOffset)
    {
        return variablesAfter[instructionOffset];
    }


    /**
     * Returns the stack before execution of the instruction at the given
     * offset.
     */
    public TracedStack getStackBefore(int instructionOffset)
    {
        return stacksBefore[instructionOffset];
    }


    /**
     * Returns the stack after execution of the instruction at the given
     * offset.
     */
    public TracedStack getStackAfter(int instructionOffset)
    {
        return stacksAfter[instructionOffset];
    }


    /**
     * Returns the instruction offsets that branch to the given instruction
     * offset.
     */
    public InstructionOffsetValue branchOrigins(int instructionOffset)
    {
        return branchOriginValues[instructionOffset];
    }


    /**
     * Returns the instruction offsets to which the given instruction offset
     * branches.
     */
    public InstructionOffsetValue branchTargets(int instructionOffset)
    {
        return branchTargetValues[instructionOffset];
    }


    /**
     * Returns a filtering version of the given instruction visitor that only
     * visits traced instructions.
     */
    public InstructionVisitor tracedInstructionFilter(InstructionVisitor instructionVisitor)
    {
        return tracedInstructionFilter(true, instructionVisitor);
    }


    /**
     * Returns a filtering version of the given instruction visitor that only
     * visits traced or untraced instructions.
     */
    public InstructionVisitor tracedInstructionFilter(boolean            traced,
                                                      InstructionVisitor instructionVisitor)
    {
        return new MyTracedInstructionFilter(traced, instructionVisitor);
    }


    // Utility methods to evaluate instruction blocks.

    /**
     * Pushes block of instructions to be executed in the calling partial
     * evaluator.
     */
    private void pushCallingInstructionBlock(TracedVariables variables,
                                             TracedStack     stack,
                                             int             startOffset)
    {
        callingInstructionBlockStack.push(new MyInstructionBlock(variables,
                                                                 stack,
                                                                 startOffset));
    }


    /**
     * Pushes block of instructions to be executed in this partial evaluator.
     */
    private void pushInstructionBlock(TracedVariables variables,
                                      TracedStack     stack,
                                      int             startOffset)
    {
        instructionBlockStack.push(new MyInstructionBlock(variables,
                                                          stack,
                                                          startOffset));
    }


    /**
     * Evaluates the instruction block and the exception handlers covering the
     * given instruction range in the given code.
     */
    private void evaluateInstructionBlockAndExceptionHandlers(Clazz           clazz,
                                                              Method          method,
                                                              CodeAttribute   codeAttribute,
                                                              TracedVariables variables,
                                                              TracedStack     stack,
                                                              int             startOffset,
                                                              int             endOffset)
    {
        evaluateInstructionBlock(clazz,
                                 method,
                                 codeAttribute,
                                 variables,
                                 stack,
                                 startOffset);

        evaluateExceptionHandlers(clazz,
                                  method,
                                  codeAttribute,
                                  startOffset,
                                  endOffset);
    }


    /**
     * Evaluates a block of instructions, starting at the given offset and ending
     * at a branch instruction, a return instruction, or a throw instruction.
     */
    private void evaluateInstructionBlock(Clazz           clazz,
                                          Method          method,
                                          CodeAttribute   codeAttribute,
                                          TracedVariables variables,
                                          TracedStack     stack,
                                          int             startOffset)
    {
        // Execute the initial instruction block.
        evaluateSingleInstructionBlock(clazz,
                                       method,
                                       codeAttribute,
                                       variables,
                                       stack,
                                       startOffset);

        // Execute all resulting instruction blocks on the execution stack.
        while (!instructionBlockStack.empty())
        {
            if (DEBUG) System.out.println("Popping alternative branch out of "+instructionBlockStack.size()+" blocks");

            MyInstructionBlock instructionBlock =
                (MyInstructionBlock)instructionBlockStack.pop();

            evaluateSingleInstructionBlock(clazz,
                                           method,
                                           codeAttribute,
                                           instructionBlock.variables,
                                           instructionBlock.stack,
                                           instructionBlock.startOffset);
        }
    }


    /**
     * Evaluates a block of instructions, starting at the given offset and ending
     * at a branch instruction, a return instruction, or a throw instruction.
     * Instruction blocks that are to be evaluated as a result are pushed on
     * the given stack.
     */
    private void evaluateSingleInstructionBlock(Clazz            clazz,
                                                Method           method,
                                                CodeAttribute    codeAttribute,
                                                TracedVariables  variables,
                                                TracedStack      stack,
                                                int              startOffset)
    {
        byte[] code = codeAttribute.code;

        if (DEBUG)
        {
             System.out.println("Instruction block starting at ["+startOffset+"] in "+
                                ClassUtil.externalFullMethodDescription(clazz.getName(),
                                                                        0,
                                                                        method.getName(clazz),
                                                                        method.getDescriptor(clazz)));
             System.out.println("Init vars:  "+variables);
             System.out.println("Init stack: "+stack);
        }

        Processor processor = new Processor(variables,
                                            stack,
                                            valueFactory,
                                            branchUnit,
                                            invocationUnit,
                                            evaluateAllCode);

        int instructionOffset = startOffset;

        int maxOffset = startOffset;

        // Evaluate the subsequent instructions.
        while (true)
        {
            if (maxOffset < instructionOffset)
            {
                maxOffset = instructionOffset;
            }

            // Maintain a generalized local variable frame and stack at this
            // instruction offset, before execution.
            int evaluationCount = evaluationCounts[instructionOffset];
            if (evaluationCount == 0)
            {
                // First time we're passing by this instruction.
                if (variablesBefore[instructionOffset] == null)
                {
                    // There's not even a context at this index yet.
                    variablesBefore[instructionOffset] = new TracedVariables(variables);
                    stacksBefore[instructionOffset]    = new TracedStack(stack);
                }
                else
                {
                    // Reuse the context objects at this index.
                    variablesBefore[instructionOffset].initialize(variables);
                    stacksBefore[instructionOffset].copy(stack);
                }

                // We'll execute in the generalized context, because it is
                // the same as the current context.
                generalizedContexts[instructionOffset] = true;
            }
            else
            {
                // Merge in the current context.
                boolean variablesChanged = variablesBefore[instructionOffset].generalize(variables, true);
                boolean stackChanged     = stacksBefore[instructionOffset].generalize(stack);

                //System.out.println("GVars:  "+variablesBefore[instructionOffset]);
                //System.out.println("GStack: "+stacksBefore[instructionOffset]);

                // Bail out if the current context is the same as last time.
                if (!variablesChanged &&
                    !stackChanged     &&
                    generalizedContexts[instructionOffset])
                {
                    if (DEBUG) System.out.println("Repeated variables, stack, and branch targets");

                    break;
                }

                // See if this instruction has been evaluated an excessive number
                // of times.
                if (evaluationCount >= MAXIMUM_EVALUATION_COUNT)
                {
                    if (DEBUG) System.out.println("Generalizing current context after "+evaluationCount+" evaluations");

                    // Continue, but generalize the current context.
                    // Note that the most recent variable values have to remain
                    // last in the generalizations, for the sake of the ret
                    // instruction.
                    variables.generalize(variablesBefore[instructionOffset], false);
                    stack.generalize(stacksBefore[instructionOffset]);

                    // We'll execute in the generalized context.
                    generalizedContexts[instructionOffset] = true;
                }
                else
                {
                    // We'll execute in the current context.
                    generalizedContexts[instructionOffset] = false;
                }
            }

            // We'll evaluate this instruction.
            evaluationCounts[instructionOffset]++;

            // Remember this instruction's offset with any stored value.
            Value storeValue = new InstructionOffsetValue(instructionOffset);
            variables.setProducerValue(storeValue);
            stack.setProducerValue(storeValue);

            // Decode the instruction.
            Instruction instruction = InstructionFactory.create(code, instructionOffset);

            // Reset the branch unit.
            branchUnit.reset();

            if (DEBUG)
            {
                System.out.println(instruction.toString(instructionOffset));
            }

            if (extraInstructionVisitor != null)
            {
                // Visit the instruction with the optional visitor.
                instruction.accept(clazz,
                                   method,
                                   codeAttribute,
                                   instructionOffset,
                                   extraInstructionVisitor);
            }

            try
            {
                // Process the instruction. The processor may modify the
                // variables and the stack, and it may call the branch unit
                // and the invocation unit.
                instruction.accept(clazz,
                                   method,
                                   codeAttribute,
                                   instructionOffset,
                                   processor);
            }
            catch (RuntimeException ex)
            {
                System.err.println("Unexpected error while evaluating instruction:");
                System.err.println("  Class       = ["+clazz.getName()+"]");
                System.err.println("  Method      = ["+method.getName(clazz)+method.getDescriptor(clazz)+"]");
                System.err.println("  Instruction = "+instruction.toString(instructionOffset));
                System.err.println("  Exception   = ["+ex.getClass().getName()+"] ("+ex.getMessage()+")");

                throw ex;
            }

            // Collect the branch targets from the branch unit.
            InstructionOffsetValue branchTargets = branchUnit.getTraceBranchTargets();
            int branchTargetCount = branchTargets.instructionOffsetCount();

            if (DEBUG)
            {
                if (branchUnit.wasCalled())
                {
                    System.out.println("     is branching to "+branchTargets);
                }
                if (branchTargetValues[instructionOffset] != null)
                {
                    System.out.println("     has up till now been branching to "+branchTargetValues[instructionOffset]);
                }

                System.out.println(" Vars:  "+variables);
                System.out.println(" Stack: "+stack);
            }

            // Maintain a generalized local variable frame and stack at this
            // instruction offset, after execution.
            if (evaluationCount == 0)
            {
                // First time we're passing by this instruction.
                if (variablesAfter[instructionOffset] == null)
                {
                    // There's not even a context at this index yet.
                    variablesAfter[instructionOffset] = new TracedVariables(variables);
                    stacksAfter[instructionOffset]    = new TracedStack(stack);
                }
                else
                {
                    // Reuse the context objects at this index.
                    variablesAfter[instructionOffset].initialize(variables);
                    stacksAfter[instructionOffset].copy(stack);
                }
            }
            else
            {
                // Merge in the current context.
                variablesAfter[instructionOffset].generalize(variables, true);
                stacksAfter[instructionOffset].generalize(stack);
            }

            // Did the branch unit get called?
            if (branchUnit.wasCalled())
            {
                // Accumulate the branch targets at this offset.
                branchTargetValues[instructionOffset] = branchTargetValues[instructionOffset] == null ?
                    branchTargets :
                    branchTargetValues[instructionOffset].generalize(branchTargets);

                // Are there no branch targets at all?
                if (branchTargetCount == 0)
                {
                    // Exit from this code block.
                    break;
                }

                // Accumulate the branch origins at the branch target offsets.
                InstructionOffsetValue instructionOffsetValue = new InstructionOffsetValue(instructionOffset);
                for (int index = 0; index < branchTargetCount; index++)
                {
                    int branchTarget = branchTargets.instructionOffset(index);
                    branchOriginValues[branchTarget] = branchOriginValues[branchTarget] == null ?
                        instructionOffsetValue:
                        branchOriginValues[branchTarget].generalize(instructionOffsetValue);
                }

                // Are there multiple branch targets?
                if (branchTargetCount > 1)
                {
                    // Push them on the execution stack and exit from this block.
                    for (int index = 0; index < branchTargetCount; index++)
                    {
                        if (DEBUG) System.out.println("Pushing alternative branch #"+index+" out of "+branchTargetCount+", from ["+instructionOffset+"] to ["+branchTargets.instructionOffset(index)+"]");

                        pushInstructionBlock(new TracedVariables(variables),
                                             new TracedStack(stack),
                                             branchTargets.instructionOffset(index));
                    }

                    break;
                }

                if (DEBUG) System.out.println("Definite branch from ["+instructionOffset+"] to ["+branchTargets.instructionOffset(0)+"]");

                // Continue at the definite branch target.
                instructionOffset = branchTargets.instructionOffset(0);
            }
            else
            {
                // Just continue with the next instruction.
                instructionOffset += instruction.length(instructionOffset);
            }

            // Is this a subroutine invocation?
            if (instruction.opcode == InstructionConstants.OP_JSR ||
                instruction.opcode == InstructionConstants.OP_JSR_W)
            {
                // Evaluate the subroutine in another partial evaluator.
                evaluateSubroutine(clazz,
                                   method,
                                   codeAttribute,
                                   variables,
                                   stack,
                                   instructionOffset);

                break;
            }
            else if (instruction.opcode == InstructionConstants.OP_RET)
            {
                // Let the partial evaluator that has called the subroutine
                // handle the evaluation after the return.
                pushCallingInstructionBlock(new TracedVariables(variables),
                                            new TracedStack(stack),
                                            instructionOffset);
                break;
            }
        }

        if (DEBUG) System.out.println("Ending processing of instruction block starting at ["+startOffset+"]");
    }


    /**
     * Evaluates a subroutine and its exception handlers, starting at the given
     * offset and ending at a subroutine return instruction.
     */
    private void evaluateSubroutine(Clazz           clazz,
                                    Method          method,
                                    CodeAttribute   codeAttribute,
                                    TracedVariables variables,
                                    TracedStack     stack,
                                    int             subroutineStart)
    {
        int subroutineEnd = branchTargetFinder.subroutineEnd(subroutineStart);

        if (DEBUG) System.out.println("Evaluating subroutine from "+subroutineStart+" to "+subroutineEnd);

        // Create a temporary partial evaluator, so there are no conflicts
        // with variables that are alive across subroutine invocations, between
        // different invocations.
        PartialEvaluator subroutinePartialEvaluator =
            new PartialEvaluator(this);

        subroutinePartialEvaluator.initializeArrays(codeAttribute);

        // Evaluate the subroutine.
        subroutinePartialEvaluator.evaluateInstructionBlockAndExceptionHandlers(clazz,
                                                                                method,
                                                                                codeAttribute,
                                                                                variables,
                                                                                stack,
                                                                                subroutineStart,
                                                                                subroutineEnd);

        // Merge back the temporary partial evaluator. This way, we'll get
        // the lowest common denominator of stacks and variables.
        generalize(subroutinePartialEvaluator, 0, codeAttribute.u4codeLength);

        if (DEBUG) System.out.println("Ending subroutine from "+subroutineStart+" to "+subroutineEnd);
    }


    /**
     * Generalizes the results of this partial evaluator with those of another
     * given partial evaluator, over a given range of instructions.
     */
    private void generalize(PartialEvaluator other,
                            int              codeStart,
                            int              codeEnd)
    {
        if (DEBUG) System.out.println("Generalizing with temporary partial evaluation");

        for (int offset = codeStart; offset < codeEnd; offset++)
        {
            if (other.branchOriginValues[offset] != null)
            {
                branchOriginValues[offset] = branchOriginValues[offset] == null ?
                    other.branchOriginValues[offset] :
                    branchOriginValues[offset].generalize(other.branchOriginValues[offset]);
            }

            if (other.isTraced(offset))
            {
                if (other.branchTargetValues[offset] != null)
                {
                    branchTargetValues[offset] = branchTargetValues[offset] == null ?
                        other.branchTargetValues[offset] :
                        branchTargetValues[offset].generalize(other.branchTargetValues[offset]);
                }

                if (evaluationCounts[offset] == 0)
                {
                    variablesBefore[offset]     = other.variablesBefore[offset];
                    stacksBefore[offset]        = other.stacksBefore[offset];
                    variablesAfter[offset]      = other.variablesAfter[offset];
                    stacksAfter[offset]         = other.stacksAfter[offset];
                    generalizedContexts[offset] = other.generalizedContexts[offset];
                    evaluationCounts[offset]    = other.evaluationCounts[offset];
                }
                else
                {
                    variablesBefore[offset].generalize(other.variablesBefore[offset], false);
                    stacksBefore[offset]   .generalize(other.stacksBefore[offset]);
                    variablesAfter[offset] .generalize(other.variablesAfter[offset], false);
                    stacksAfter[offset]    .generalize(other.stacksAfter[offset]);
                    //generalizedContexts[offset]
                    evaluationCounts[offset] += other.evaluationCounts[offset];
                }
            }
        }
    }


    /**
     * Evaluates the exception handlers covering and targeting the given
     * instruction range in the given code.
     */
    private void evaluateExceptionHandlers(Clazz         clazz,
                                           Method        method,
                                           CodeAttribute codeAttribute,
                                           int           startOffset,
                                           int           endOffset)
    {
        if (DEBUG) System.out.println("Evaluating exceptions covering ["+startOffset+" -> "+endOffset+"]:");

        ExceptionHandlerFilter exceptionEvaluator =
            new ExceptionHandlerFilter(startOffset,
                                       endOffset,
                                       this);

        // Evaluate the exception catch blocks, until their entry variables
        // have stabilized.
        do
        {
            // Reset the flag to stop evaluating.
            evaluateExceptions = false;

            // Evaluate all relevant exception catch blocks once.
            codeAttribute.exceptionsAccept(clazz,
                                           method,
                                           startOffset,
                                           endOffset,
                                           exceptionEvaluator);
        }
        while (evaluateExceptions);
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        int startPC = exceptionInfo.u2startPC;
        int endPC   = exceptionInfo.u2endPC;

        // Do we have to evaluate this exception catch block?
        if (mayThrowExceptions(clazz, method, codeAttribute, startPC, endPC))
        {
            int handlerPC = exceptionInfo.u2handlerPC;
            int catchType = exceptionInfo.u2catchType;

            if (DEBUG) System.out.println("Evaluating exception ["+startPC +" -> "+endPC +": "+handlerPC+"]:");

            // Reuse the existing variables and stack objects, ensuring the
            // right size.
            TracedVariables variables = new TracedVariables(codeAttribute.u2maxLocals);
            TracedStack     stack     = new TracedStack(codeAttribute.u2maxStack);

            // Initialize the trace values.
            Value storeValue = new InstructionOffsetValue(handlerPC | InstructionOffsetValue.EXCEPTION_HANDLER);
            variables.setProducerValue(storeValue);
            stack.setProducerValue(storeValue);

            // Initialize the variables by generalizing the variables of the
            // try block. Make sure to include the results of the last
            // instruction for preverification.
            generalizeVariables(startPC,
                                endPC,
                                evaluateAllCode,
                                variables);

            // Initialize the stack.
            invocationUnit.enterExceptionHandler(clazz,
                                                 method,
                                                 codeAttribute,
                                                 handlerPC,
                                                 catchType,
                                                 stack);

            int evaluationCount = evaluationCounts[handlerPC];

            // Evaluate the instructions, starting at the entry point.
            evaluateInstructionBlock(clazz,
                                     method,
                                     codeAttribute,
                                     variables,
                                     stack,
                                     handlerPC);

            // Remember to evaluate all exception handlers once more.
            if (!evaluateExceptions)
            {
                evaluateExceptions = evaluationCount < evaluationCounts[handlerPC];
            }
        }
//        else if (evaluateAllCode)
//        {
//            if (DEBUG) System.out.println("No information for partial evaluation of exception ["+startPC +" -> "+endPC +": "+exceptionInfo.u2handlerPC+"] yet");
//
//            // We don't have any information on the try block yet, but we do
//            // have to evaluate the exception handler.
//            // Remember to evaluate all exception handlers once more.
//            evaluateExceptions = true;
//        }
        else
        {
            if (DEBUG) System.out.println("No information for partial evaluation of exception ["+startPC +" -> "+endPC +": "+exceptionInfo.u2handlerPC+"]");
        }
    }


    // Small utility methods.

    /**
     * Initializes the data structures for the variables, stack, etc.
     */
    private void initializeArrays(CodeAttribute codeAttribute)
    {
        int newCodeLength = codeAttribute.u4codeLength;

        // Create new arrays for storing information at each instruction offset.
        if (branchOriginValues.length < newCodeLength)
        {
            // Create new arrays.
            branchOriginValues  = new InstructionOffsetValue[newCodeLength];
            branchTargetValues  = new InstructionOffsetValue[newCodeLength];
            variablesBefore     = new TracedVariables[newCodeLength];
            stacksBefore        = new TracedStack[newCodeLength];
            variablesAfter      = new TracedVariables[newCodeLength];
            stacksAfter         = new TracedStack[newCodeLength];
            generalizedContexts = new boolean[newCodeLength];
            evaluationCounts    = new int[newCodeLength];
        }
        else
        {
            // Reset the old arrays.
            Arrays.fill(branchOriginValues,  0, codeLength, null);
            Arrays.fill(branchTargetValues,  0, codeLength, null);
            Arrays.fill(generalizedContexts, 0, codeLength, false);
            Arrays.fill(evaluationCounts,    0, codeLength, 0);

            for (int index = 0; index < newCodeLength; index++)
            {
                if (variablesBefore[index] != null)
                {
                    variablesBefore[index].reset(codeAttribute.u2maxLocals);
                }

                if (stacksBefore[index] != null)
                {
                    stacksBefore[index].reset(codeAttribute.u2maxStack);
                }

                if (variablesAfter[index] != null)
                {
                    variablesAfter[index].reset(codeAttribute.u2maxLocals);
                }

                if (stacksAfter[index] != null)
                {
                    stacksAfter[index].reset(codeAttribute.u2maxStack);
                }
            }

            for (int index = newCodeLength; index < codeLength; index++)
            {
                if (variablesBefore[index] != null)
                {
                    variablesBefore[index].reset(0);
                }

                if (stacksBefore[index] != null)
                {
                    stacksBefore[index].reset(0);
                }

                if (variablesAfter[index] != null)
                {
                    variablesAfter[index].reset(0);
                }

                if (stacksAfter[index] != null)
                {
                    stacksAfter[index].reset(0);
                }
            }
        }

        codeLength = newCodeLength;
    }


    /**
     * Initializes the data structures for the variables, stack, etc.
     */
    private void initializeParameters(Clazz           clazz,
                                      Method          method,
                                      CodeAttribute   codeAttribute,
                                      TracedVariables variables)
    {
//        // Create the method parameters.
//        TracedVariables parameters = new TracedVariables(codeAttribute.u2maxLocals);
//
//        // Remember this instruction's offset with any stored value.
//        Value storeValue = new InstructionOffsetValue(AT_METHOD_ENTRY);
//        parameters.setProducerValue(storeValue);

        // Create the method parameters.
        Variables parameters = new Variables(codeAttribute.u2maxLocals);

        // Initialize the method parameters.
        invocationUnit.enterMethod(clazz, method, parameters);

        if (DEBUG)
        {
            System.out.println("  Params: "+parameters);
        }

        // Initialize the variables with the parameters.
        variables.initialize(parameters);

        // Set the store value of each parameter variable. We store the
        // variable indices of the parameters. These parameter offsets take
        // into account Category 2 types.
        for (int index = 0; index < parameters.size(); index++)
        {
            InstructionOffsetValue producerValue =
                new InstructionOffsetValue(index | InstructionOffsetValue.METHOD_PARAMETER);

            variables.setProducerValue(index, producerValue);
        }
    }


    /**
     * Returns whether a block of instructions may ever throw an exception.
     */
    private boolean mayThrowExceptions(Clazz         clazz,
                                       Method        method,
                                       CodeAttribute codeAttribute,
                                       int           startOffset,
                                       int           endOffset)
    {
        for (int index = startOffset; index < endOffset; index++)
        {
            if (isTraced(index) &&
                (evaluateAllCode ||
                 InstructionFactory.create(codeAttribute.code, index).mayThrowExceptions()))
            {
                return true;
            }
        }

        return false;
    }


    /**
     * Generalize the local variable frames of a block of instructions.
     */
    private void generalizeVariables(int             startOffset,
                                     int             endOffset,
                                     boolean         includeAfterLastInstruction,
                                     TracedVariables generalizedVariables)
    {
        boolean first     = true;
        int     lastIndex = -1;

        // Generalize the variables before each of the instructions in the block.
        for (int index = startOffset; index < endOffset; index++)
        {
            if (isTraced(index))
            {
                TracedVariables tracedVariables = variablesBefore[index];

                if (first)
                {
                    // Initialize the variables with the first traced local
                    // variable frame.
                    generalizedVariables.initialize(tracedVariables);

                    first = false;
                }
                else
                {
                    // Generalize the variables with the traced local variable
                    // frame. We can't use the return value, because local
                    // generalization can be different a couple of times,
                    // with the global generalization being the same.
                    generalizedVariables.generalize(tracedVariables, false);
                }

                lastIndex = index;
            }
        }

        // Generalize the variables after the last instruction in the block,
        // if required.
        if (includeAfterLastInstruction &&
            lastIndex >= 0)
        {
            TracedVariables tracedVariables = variablesAfter[lastIndex];

            if (first)
            {
                // Initialize the variables with the local variable frame.
                generalizedVariables.initialize(tracedVariables);
            }
            else
            {
                // Generalize the variables with the local variable frame.
                generalizedVariables.generalize(tracedVariables, false);
            }
        }

        // Just clear the variables if there aren't any traced instructions
        // in the block.
        if (first)
        {
            generalizedVariables.reset(generalizedVariables.size());
        }
    }


    /**
     * This class represents an instruction block that has to be executed,
     * starting with a given state at a given instruction offset.
     */
    private static class MyInstructionBlock
    {
        private TracedVariables variables;
        private TracedStack     stack;
        private int             startOffset;


        private MyInstructionBlock(TracedVariables variables,
                                   TracedStack     stack,
                                   int             startOffset)
        {
            this.variables   = variables;
            this.stack       = stack;
            this.startOffset = startOffset;
        }
    }


   /**
     * This InstructionVisitor delegates its visits to a given
     * InstructionVisitor, but only if the instruction has been traced (or not).
     */
    private class MyTracedInstructionFilter implements InstructionVisitor
    {
        private final boolean            traced;
        private final InstructionVisitor instructionVisitor;


        public MyTracedInstructionFilter(boolean            traced,
                                            InstructionVisitor instructionVisitor)
        {
            this.traced          = traced;
            this.instructionVisitor = instructionVisitor;
        }


        // Implementations for InstructionVisitor.

        public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
        {
            if (shouldVisit(offset))
            {
                instructionVisitor.visitSimpleInstruction(clazz, method, codeAttribute, offset, simpleInstruction);
            }
        }


        public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
        {
            if (shouldVisit(offset))
            {
                instructionVisitor.visitVariableInstruction(clazz, method, codeAttribute, offset, variableInstruction);
            }
        }


        public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
        {
            if (shouldVisit(offset))
            {
                instructionVisitor.visitConstantInstruction(clazz, method, codeAttribute, offset, constantInstruction);
            }
        }


        public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
        {
            if (shouldVisit(offset))
            {
                instructionVisitor.visitBranchInstruction(clazz, method, codeAttribute, offset, branchInstruction);
            }
        }


        public void visitTableSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, TableSwitchInstruction tableSwitchInstruction)
        {
            if (shouldVisit(offset))
            {
                instructionVisitor.visitTableSwitchInstruction(clazz, method, codeAttribute, offset, tableSwitchInstruction);
            }
        }


        public void visitLookUpSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LookUpSwitchInstruction lookUpSwitchInstruction)
        {
            if (shouldVisit(offset))
            {
                instructionVisitor.visitLookUpSwitchInstruction(clazz, method, codeAttribute, offset, lookUpSwitchInstruction);
            }
        }


        private boolean shouldVisit(int offset)
        {
            return isTraced(offset) == traced;
        }
   }
}
