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
import proguard.classfile.constant.RefConstant;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.CodeAttributeEditor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;
import proguard.evaluation.TracedStack;
import proguard.evaluation.value.*;
import proguard.optimize.info.ParameterUsageMarker;

/**
 * This AttributeVisitor shrinks the code attributes that it visits, based
 * on partial evaluation.
 *
 * @author Eric Lafortune
 */
public class EvaluationShrinker
extends      SimplifiedVisitor
implements   AttributeVisitor,
             ExceptionInfoVisitor
{
    //*
    private static final boolean DEBUG          = false;
    private static final boolean DEBUG_RESULTS  = false;
    /*/
    private static boolean DEBUG          = System.getProperty("es") != null;
    private static boolean DEBUG_RESULTS  = DEBUG;
    //*/

    private static final int UNSUPPORTED         = -1;
    private static final int NOP                 = InstructionConstants.OP_NOP     & 0xff;
    private static final int POP                 = InstructionConstants.OP_POP     & 0xff;
    private static final int POP2                = InstructionConstants.OP_POP2    & 0xff;
    private static final int DUP                 = InstructionConstants.OP_DUP     & 0xff;
    private static final int DUP_X1              = InstructionConstants.OP_DUP_X1  & 0xff;
    private static final int DUP_X2              = InstructionConstants.OP_DUP_X2  & 0xff;
    private static final int DUP2                = InstructionConstants.OP_DUP2    & 0xff;
    private static final int DUP2_X1             = InstructionConstants.OP_DUP2_X1 & 0xff;
    private static final int DUP2_X2             = InstructionConstants.OP_DUP2_X2 & 0xff;
    private static final int SWAP                = InstructionConstants.OP_SWAP    & 0xff;
    private static final int MOV_X2              = DUP_X2  | (POP     << 8);
    private static final int MOV2_X1             = DUP2_X1 | (POP2    << 8);
    private static final int MOV2_X2             = DUP2_X2 | (POP2    << 8);
    private static final int POP_X1              = SWAP    | (POP     << 8);
    private static final int POP_X2              = DUP2_X1 | (POP2    << 8) | (POP    << 16);
    private static final int POP_X3              = UNSUPPORTED;
    private static final int POP2_X1             = DUP_X2  | (POP     << 8) | (POP2   << 16);
    private static final int POP2_X2             = DUP2_X2 | (POP2    << 8) | (POP2   << 16);
    private static final int POP3                = POP2    | (POP     << 8);
    private static final int POP4                = POP2    | (POP2    << 8);
    private static final int POP_DUP             = POP     | (DUP     << 8);
    private static final int POP_DUP_X1          = POP     | (DUP_X1  << 8);
    private static final int POP_SWAP            = POP     | (SWAP    << 8);
    private static final int POP_SWAP_POP        = POP     | (SWAP    << 8) | (POP    << 16);
    private static final int POP_SWAP_POP_DUP    = POP     | (SWAP    << 8) | (POP    << 16) | (DUP    << 24);
    private static final int POP2_SWAP_POP       = POP2    | (SWAP    << 8) | (POP    << 16);
    private static final int SWAP_DUP_X1         = SWAP    | (DUP_X1  << 8);
    private static final int SWAP_DUP_X1_POP3    = SWAP    | (DUP_X1  << 8) | (POP2   << 16) | (POP    << 24);
    private static final int SWAP_DUP2_X1_POP3   = SWAP    | (DUP2_X1 << 8) | (POP2   << 16) | (POP    << 24);
    private static final int SWAP_DUP_X1_SWAP    = SWAP    | (DUP_X1  << 8) | (SWAP   << 16);
    private static final int SWAP_POP_DUP        = SWAP    | (POP     << 8) | (DUP    << 16);
    private static final int SWAP_POP_DUP_X1     = SWAP    | (POP     << 8) | (DUP_X1 << 16);
    private static final int DUP_X2_POP          = DUP_X2  | (POP     << 8);
    private static final int DUP_X2_POP2         = DUP_X2  | (POP2    << 8);
    private static final int DUP_X2_POP3_DUP     = DUP_X2  | (POP     << 8) | (POP2   << 16) | (DUP    << 24);
    private static final int DUP2_X1_POP         = DUP2_X1 | (POP     << 8);
    private static final int DUP2_X1_POP3_DUP    = DUP2_X1 | (POP2    << 8) | (POP    << 16) | (DUP    << 24);
    private static final int DUP2_X1_POP3_DUP_X1 = DUP2_X1 | (POP2    << 8) | (POP    << 16) | (DUP_X1 << 24);
    private static final int DUP2_X1_POP3_DUP2   = DUP2_X1 | (POP2    << 8) | (POP    << 16) | (DUP2    << 24);
    private static final int DUP2_X2_POP3        = DUP2_X2 | (POP2    << 8) | (POP    << 16);
    private static final int DUP2_X2_SWAP_POP    = DUP2_X2 | (SWAP    << 8) | (POP    << 16);


    private final InstructionUsageMarker instructionUsageMarker;
    private final boolean                runInstructionUsageMarker;
    private final InstructionVisitor     extraDeletedInstructionVisitor;
    private final InstructionVisitor     extraAddedInstructionVisitor;

    private final MyStaticInvocationFixer       staticInvocationFixer       = new MyStaticInvocationFixer();
    private final MyBackwardBranchFixer         backwardBranchFixer         = new MyBackwardBranchFixer();
    private final MyNonReturningSubroutineFixer nonReturningSubroutineFixer = new MyNonReturningSubroutineFixer();
    private final MyStackConsistencyFixer       stackConsistencyFixer       = new MyStackConsistencyFixer();
    private final MyInstructionDeleter          instructionDeleter          = new MyInstructionDeleter();
    private final CodeAttributeEditor           codeAttributeEditor         = new CodeAttributeEditor(false, true);


    /**
     * Creates a new EvaluationShrinker.
     */
    public EvaluationShrinker()
    {
        this(new PartialEvaluator(), true, null, null);
    }


    /**
     * Creates a new EvaluationShrinker.
     * @param partialEvaluator               the partial evaluator that will
     *                                       analyze the code.
     * @param runPartialEvaluator            specifies whether the partial
     *                                       evaluator should be run for each
     *                                       method, or if some other class is
     *                                       already doing this.
     * @param extraDeletedInstructionVisitor an optional extra visitor for all
     *                                       deleted instructions.
     * @param extraAddedInstructionVisitor   an optional extra visitor for all
     *                                       added instructions.
     */
    public EvaluationShrinker(PartialEvaluator   partialEvaluator,
                              boolean            runPartialEvaluator,
                              InstructionVisitor extraDeletedInstructionVisitor,
                              InstructionVisitor extraAddedInstructionVisitor)
    {
        this(new InstructionUsageMarker(partialEvaluator, runPartialEvaluator),
             true,
             extraDeletedInstructionVisitor,
             extraAddedInstructionVisitor);
    }


    /**
     * Creates a new EvaluationShrinker.
     * @param instructionUsageMarker         the instruction usage marker that
     *                                       will analyze the code.
     * @param runInstructionUsageMarker      specifies whether the usage
     *                                       marker should be run for each
     *                                       method, or if some other class is
     *                                       already doing this.
     * @param extraDeletedInstructionVisitor an optional extra visitor for all
     *                                       deleted instructions.
     * @param extraAddedInstructionVisitor   an optional extra visitor for all
     *                                       added instructions.
     */
    public EvaluationShrinker(InstructionUsageMarker instructionUsageMarker,
                              boolean                runInstructionUsageMarker,
                              InstructionVisitor     extraDeletedInstructionVisitor,
                              InstructionVisitor     extraAddedInstructionVisitor)
    {
        this.instructionUsageMarker         = instructionUsageMarker;
        this.runInstructionUsageMarker      = runInstructionUsageMarker;
        this.extraDeletedInstructionVisitor = extraDeletedInstructionVisitor;
        this.extraAddedInstructionVisitor   = extraAddedInstructionVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
//        DEBUG = DEBUG_RESULTS =
//            clazz.getName().equals("abc/Def") &&
//            method.getName(clazz).equals("abc");

        // TODO: Remove this when the evaluation shrinker has stabilized.
        // Catch any unexpected exceptions from the actual visiting method.
        try
        {
            // Process the code.
            visitCodeAttribute0(clazz, method, codeAttribute);
        }
        catch (RuntimeException ex)
        {
            System.err.println("Unexpected error while shrinking instructions after partial evaluation:");
            System.err.println("  Class       = ["+clazz.getName()+"]");
            System.err.println("  Method      = ["+method.getName(clazz)+method.getDescriptor(clazz)+"]");
            System.err.println("  Exception   = ["+ex.getClass().getName()+"] ("+ex.getMessage()+")");

            ex.printStackTrace();
            System.err.println("Not optimizing this method");

            if (DEBUG)
            {
                method.accept(clazz, new ClassPrinter());

                throw ex;
            }
        }
    }


    public void visitCodeAttribute0(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        if (DEBUG_RESULTS)
        {
            System.out.println("EvaluationShrinker ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]");
        }

        // Analyze the method.
        if (runInstructionUsageMarker)
        {
            instructionUsageMarker.visitCodeAttribute(clazz, method, codeAttribute);
        }

        int codeLength = codeAttribute.u4codeLength;

        if (DEBUG) System.out.println();


        // Reset the code changes.
        codeAttributeEditor.reset(codeLength);

        // Replace virtual invocations by static invocations, where necessary.
        if (DEBUG) System.out.println("Static invocation fixing:");

        codeAttribute.instructionsAccept(clazz, method,
            instructionUsageMarker.necessaryInstructionFilter(true,
            staticInvocationFixer));

        if (DEBUG) System.out.println();


        // Replace traced but unnecessary backward branches by infinite loops.
        // The virtual machine's verification step is not smart enough to see
        // the code isn't reachable, and may complain otherwise.
        // Any clearly unreachable code will still be removed elsewhere.
        if (DEBUG) System.out.println("Backward branch fixing:");

        codeAttribute.instructionsAccept(clazz, method,
            instructionUsageMarker.tracedInstructionFilter(true,
            instructionUsageMarker.necessaryInstructionFilter(false,
            backwardBranchFixer)));

        if (DEBUG) System.out.println();


        // Insert infinite loops after jumps to subroutines that don't return.
        // The virtual machine's verification step is not smart enough to see
        // the code isn't reachable, and may complain otherwise.
        if (DEBUG) System.out.println("Non-returning subroutine fixing:");

        codeAttribute.instructionsAccept(clazz, method,
            instructionUsageMarker.necessaryInstructionFilter(true,
            nonReturningSubroutineFixer));

        if (DEBUG) System.out.println();


        // Locally fix instructions, in order to keep the stack consistent.
        if (DEBUG) System.out.println("Stack consistency fixing:");

        codeAttribute.instructionsAccept(clazz, method,
            instructionUsageMarker.tracedInstructionFilter(true,
            stackConsistencyFixer));

        if (DEBUG) System.out.println();


        // Delete all instructions that are not used.
        if (DEBUG) System.out.println("Deleting unused instructions");

        codeAttribute.instructionsAccept(clazz, method,
            instructionUsageMarker.necessaryInstructionFilter(false,
            instructionDeleter));

        if (DEBUG) System.out.println();


        if (DEBUG_RESULTS)
        {
            System.out.println("Simplification results:");

            int offset = 0;
            do
            {
                Instruction instruction = InstructionFactory.create(codeAttribute.code,
                                                                    offset);
                System.out.println((instructionUsageMarker.isInstructionNecessary(offset)             ? " + " :
                                    instructionUsageMarker.isExtraPushPopInstructionNecessary(offset) ? " ~ " :
                                                                                                        " - ") +
                                   instruction.toString(offset));

                if (instructionUsageMarker.isTraced(offset))
                {
                    InstructionOffsetValue branchTargets = instructionUsageMarker.branchTargets(offset);
                    if (branchTargets != null)
                    {
                        System.out.println("     has overall been branching to "+branchTargets);
                    }

                    boolean deleted = codeAttributeEditor.deleted[offset];
                    if (instructionUsageMarker.isInstructionNecessary(offset) && deleted)
                    {
                        System.out.println("     is deleted");
                    }

                    Instruction preInsertion = codeAttributeEditor.preInsertions[offset];
                    if (preInsertion != null)
                    {
                        System.out.println("     is preceded by: "+preInsertion);
                    }

                    Instruction replacement = codeAttributeEditor.replacements[offset];
                    if (replacement != null)
                    {
                        System.out.println("     is replaced by: "+replacement);
                    }

                    Instruction postInsertion = codeAttributeEditor.postInsertions[offset];
                    if (postInsertion != null)
                    {
                        System.out.println("     is followed by: "+postInsertion);
                    }
                }

                offset += instruction.length(offset);
            }
            while (offset < codeLength);
        }

        // Clear exception handlers that are not necessary.
        codeAttribute.exceptionsAccept(clazz, method, this);

        // Apply all accumulated changes to the code.
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
    }


    /**
     * This MemberVisitor converts virtual method invocations into static
     * method invocations if the 'this' parameter isn't used.
     */
    private class MyStaticInvocationFixer
    extends       SimplifiedVisitor
    implements    InstructionVisitor,
                  ConstantVisitor,
                  MemberVisitor
    {
        private int                 invocationOffset;
        private ConstantInstruction invocationInstruction;


        // Implementations for InstructionVisitor.

        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


        public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
        {
            switch (constantInstruction.opcode)
            {
                case InstructionConstants.OP_INVOKEVIRTUAL:
                case InstructionConstants.OP_INVOKESPECIAL:
                case InstructionConstants.OP_INVOKEINTERFACE:
                    this.invocationOffset      = offset;
                    this.invocationInstruction = constantInstruction;
                    clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                    break;
            }
        }


        // Implementations for ConstantVisitor.

        public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
        {
            refConstant.referencedMemberAccept(this);
        }


        // Implementations for MemberVisitor.

        public void visitAnyMember(Clazz clazz, Member member) {}


        public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
        {
            // Make the method invocation static, if possible.
            if ((programMethod.getAccessFlags() & ClassConstants.ACC_STATIC) == 0 &&
                !ParameterUsageMarker.isParameterUsed(programMethod, 0))
            {
                replaceByStaticInvocation(programClass,
                                          invocationOffset,
                                          invocationInstruction);
            }
        }
    }


    /**
     * This InstructionVisitor replaces all backward branches by
     * infinite loops.
     */
    private class MyBackwardBranchFixer
    extends       SimplifiedVisitor
    implements    InstructionVisitor
    {
        // Implementations for InstructionVisitor.

        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
        {
            // Is it a traced but unmarked backward branch, without an unmarked
            // straddling forward branch? Note that this is still a heuristic.
            if (isAllSmallerThanOrEqual(instructionUsageMarker.branchTargets(offset),
                                        offset) &&
                !isAnyUnnecessaryInstructionBranchingOver(lastNecessaryInstructionOffset(offset),
                                                          offset))
            {
                replaceByInfiniteLoop(clazz, offset);

                if (DEBUG) System.out.println("  Setting infinite loop instead of "+instruction.toString(offset));
            }
        }


        /**
         * Returns whether all of the given instruction offsets (at least one)
         * are smaller than or equal to the given offset.
         */
        private boolean isAllSmallerThanOrEqual(InstructionOffsetValue instructionOffsets,
                                                int                    instructionOffset)
        {
            if (instructionOffsets != null)
            {
                // Loop over all instruction offsets.
                int branchCount = instructionOffsets.instructionOffsetCount();
                if (branchCount > 0)
                {
                    for (int branchIndex = 0; branchIndex < branchCount; branchIndex++)
                    {
                        // Is the offset larger than the reference offset?
                        if (instructionOffsets.instructionOffset(branchIndex) > instructionOffset)
                        {
                            return false;
                        }
                    }

                    return true;
                }
            }

            return false;
        }


        /**
         * Returns the highest offset of an instruction that has been marked as
         * necessary, before the given offset.
         */
        private int lastNecessaryInstructionOffset(int instructionOffset)
        {
            for (int offset = instructionOffset-1; offset >= 0; offset--)
            {
                if (instructionUsageMarker.isInstructionNecessary(instructionOffset))
                {
                    return offset;
                }
            }

            return 0;
        }
    }


    /**
     * This InstructionVisitor appends infinite loops after all visited
     * non-returning subroutine invocations.
     */
    private class MyNonReturningSubroutineFixer
    extends       SimplifiedVisitor
    implements    InstructionVisitor
    {
        // Implementations for InstructionVisitor.

        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


        public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
        {
            // Is it a necessary subroutine invocation?
            if (branchInstruction.canonicalOpcode() == InstructionConstants.OP_JSR)
            {
                int nextOffset = offset + branchInstruction.length(offset);
                if (!instructionUsageMarker.isInstructionNecessary(nextOffset))
                {
                    replaceByInfiniteLoop(clazz, nextOffset);

                    if (DEBUG) System.out.println("  Adding infinite loop at ["+nextOffset+"] after "+branchInstruction.toString(offset));
                }
            }
        }
    }


    /**
     * This InstructionVisitor fixes instructions locally, popping any unused
     * produced stack entries after marked instructions, and popping produced
     * stack entries and pushing missing stack entries instead of unmarked
     * instructions.
     */
    private class MyStackConsistencyFixer
    extends       SimplifiedVisitor
    implements    InstructionVisitor
    {
        // Implementations for InstructionVisitor.

        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
        {
            // Has the instruction been marked?
            if (instructionUsageMarker.isInstructionNecessary(offset))
            {
                // Check all stack entries that are popped.
                // Unusual case: an exception handler with an exception that is
                // no longer consumed as a method parameter.
                // Typical case: a freshly marked variable initialization that
                // requires some value on the stack.
                int popCount = instruction.stackPopCount(clazz);
                if (popCount > 0)
                {
                    TracedStack tracedStack =
                        instructionUsageMarker.getStackBefore(offset);

                    int stackSize = tracedStack.size();

                    int requiredPopCount  = 0;
                    int requiredPushCount = 0;
                    for (int stackIndex = stackSize - popCount; stackIndex < stackSize; stackIndex++)
                    {
                        boolean stackEntryUnwantedBefore =
                            instructionUsageMarker.isStackEntryUnwantedBefore(offset, stackIndex);
                        boolean stackEntryPresentBefore =
                            instructionUsageMarker.isStackEntryPresentBefore(offset, stackIndex);

                        if (stackEntryUnwantedBefore)
                        {
                            if (stackEntryPresentBefore)
                            {
                                // Check if it is an exception pushed by a
                                // handler (should be) that is not at the
                                // top of the stack ([PGD-748], test2162).
                                InstructionOffsetValue producers =
                                    tracedStack.getBottomProducerValue(stackIndex).instructionOffsetValue();
                                if (producers.instructionOffsetCount() == 1 &&
                                    producers.isExceptionHandler(0)         &&
                                    stackIndex < stackSize - 1)
                                {
                                    // Try to handle it.
                                    // This only works if the exception isn't
                                    // consumed elsewhere (should be ok, since
                                    // it's an unused parameter).
                                    if (DEBUG) System.out.println("  Popping exception at handler "+instruction.toString(offset));

                                    insertPopInstructions(producers.instructionOffset(0), false, true, 1);
                                }
                                else
                                {
                                    // Remember to pop it.
                                    requiredPopCount++;
                                }
                            }
                        }
                        else
                        {
                            if (!stackEntryPresentBefore)
                            {
                                // Remember to push some value.
                                requiredPushCount++;
                            }
                        }
                    }

                    // Pop some unnecessary stack entries.
                    // This only works if the entries are at the top of the stack.
                    if (requiredPopCount > 0)
                    {
                        if (DEBUG) System.out.println("  Popping "+requiredPopCount+" entries before marked consumer "+instruction.toString(offset));

                        insertPopInstructions(offset, false, true, requiredPopCount);
                    }

                    // Push some necessary stack entries.
                    // This only works if the entries are at the top of the stack.
                    if (requiredPushCount > 0)
                    {
                        Value value = tracedStack.getTop(0);

                        if (DEBUG) System.out.println("  Pushing type "+value.computationalType()+" entry before marked consumer "+instruction.toString(offset));

                        if (requiredPushCount > (value.isCategory2() ? 2 : 1))
                        {
                            throw new IllegalArgumentException("Unsupported stack size increment ["+requiredPushCount+"] at ["+offset+"]");
                        }

                        insertPushInstructions(offset, false, true, value.computationalType());
                    }
                }

                // Check all stack entries that are pushed.
                // Typical case: a return value that wasn't really required and
                // that should be popped.
                int pushCount = instruction.stackPushCount(clazz);
                if (pushCount > 0)
                {
                    TracedStack tracedStack =
                        instructionUsageMarker.getStackAfter(offset);

                    int stackSize = tracedStack.size();

                    int requiredPopCount = 0;
                    for (int stackIndex = stackSize - pushCount; stackIndex < stackSize; stackIndex++)
                    {
                        // Is the stack entry required by consumers?
                        if (!instructionUsageMarker.isStackEntryNecessaryAfter(offset, stackIndex))
                        {
                            // Remember to pop it.
                            requiredPopCount++;
                        }
                    }

                    // Pop the unnecessary stack entries.
                    if (requiredPopCount > 0)
                    {
                        if (DEBUG) System.out.println("  Popping "+requiredPopCount+" entries after marked producer "+instruction.toString(offset));

                        insertPopInstructions(offset, false, false, requiredPopCount);
                    }
                }
            }
            else if (instructionUsageMarker.isExtraPushPopInstructionNecessary(offset))
            {
                // Check all stack entries that would be popped.
                // Typical case: a stack value that is required elsewhere and
                // that still has to be popped.
                int popCount = instruction.stackPopCount(clazz);
                if (popCount > 0)
                {
                    TracedStack tracedStack =
                        instructionUsageMarker.getStackBefore(offset);

                    int stackSize = tracedStack.size();

                    int expectedPopCount = 0;
                    for (int stackIndex = stackSize - popCount; stackIndex < stackSize; stackIndex++)
                    {
                        // Is this stack entry pushed by any producer
                        // (because it is required by other consumers)?
                        if (instructionUsageMarker.isStackEntryPresentBefore(offset, stackIndex))
                        {
                            // Remember to pop it.
                            expectedPopCount++;
                        }
                    }

                    // Pop the unnecessary stack entries.
                    if (expectedPopCount > 0)
                    {
                        if (DEBUG) System.out.println("  Popping "+expectedPopCount+" entries instead of unmarked consumer "+instruction.toString(offset));

                        insertPopInstructions(offset, true, false, expectedPopCount);
                    }
                }

                // Check all stack entries that would be pushed.
                // Typical case: a corresponding stack entry is pushed
                // elsewhere so it still has to be pushed here.
                int pushCount = instruction.stackPushCount(clazz);
                if (pushCount > 0)
                {
                    TracedStack tracedStack =
                        instructionUsageMarker.getStackAfter(offset);

                    int stackSize = tracedStack.size();

                    int expectedPushCount = 0;
                    for (int stackIndex = stackSize - pushCount; stackIndex < stackSize; stackIndex++)
                    {
                        // Is the stack entry required by consumers?
                        if (instructionUsageMarker.isStackEntryNecessaryAfter(offset, stackIndex))
                        {
                            // Remember to push it.
                            expectedPushCount++;
                        }
                    }

                    // Push some necessary stack entries.
                    if (expectedPushCount > 0)
                    {
                        if (DEBUG) System.out.println("  Pushing type "+tracedStack.getTop(0).computationalType()+" entry instead of unmarked producer "+instruction.toString(offset));

                        insertPushInstructions(offset, true, false, tracedStack.getTop(0).computationalType());
                    }
                }
            }
        }


        public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
        {
            if (instructionUsageMarker.isInstructionNecessary(offset) &&
                isDupOrSwap(simpleInstruction))
            {
                int topBefore = instructionUsageMarker.getStackBefore(offset).size() - 1;
                int topAfter  = instructionUsageMarker.getStackAfter(offset).size()  - 1;

                byte oldOpcode = simpleInstruction.opcode;

                // Simplify the dup/swap instruction if possible.
                int newOpcodes = fixDupSwap(offset, oldOpcode, topBefore, topAfter);

                // Did we find a suitable (extended) opcode?
                if (newOpcodes == UNSUPPORTED)
                {
                    // We can't easily emulate some constructs.
                    throw new UnsupportedOperationException("Can't handle "+simpleInstruction.toString()+" instruction at ["+offset +"]");
                }

                // Is there a single replacement opcode?
                if ((newOpcodes & ~0xff) == 0)
                {
                    byte newOpcode = (byte)newOpcodes;

                    if      (newOpcode == InstructionConstants.OP_NOP)
                    {
                        // Delete the instruction.
                        codeAttributeEditor.deleteInstruction(offset);

                        if (extraDeletedInstructionVisitor != null)
                        {
                            extraDeletedInstructionVisitor.visitSimpleInstruction(null, null, null, offset, null);
                        }

                        if (DEBUG) System.out.println("  Deleting marked instruction "+simpleInstruction.toString(offset));
                    }
                    else if (newOpcode == oldOpcode)
                    {
                        // Leave the instruction unchanged.
                        codeAttributeEditor.undeleteInstruction(offset);

                        if (DEBUG) System.out.println("  Marking unchanged instruction "+simpleInstruction.toString(offset));
                    }
                    else
                    {
                        // Replace the instruction.
                        Instruction replacementInstruction = new SimpleInstruction(newOpcode);
                        codeAttributeEditor.replaceInstruction(offset,
                                                               replacementInstruction);

                        if (DEBUG) System.out.println("  Replacing instruction "+simpleInstruction.toString(offset)+" by "+replacementInstruction.toString());
                    }
                }
                else
                {
                    // Collect the replacement instructions.
                    Instruction[] replacementInstructions = new Instruction[4];

                    if (DEBUG) System.out.println("  Replacing instruction "+simpleInstruction.toString(offset)+" by");
                    int count = 0;
                    while (newOpcodes != 0)
                    {
                        SimpleInstruction replacementInstruction = new SimpleInstruction((byte)newOpcodes);
                        replacementInstructions[count++] = replacementInstruction;

                        if (DEBUG) System.out.println("    "+replacementInstruction.toString());
                        newOpcodes >>>= 8;
                    }

                    // Create a properly sized array.
                    if (count < 4)
                    {
                        Instruction[] newInstructions = new Instruction[count];
                        System.arraycopy(replacementInstructions, 0, newInstructions, 0, count);
                        replacementInstructions = newInstructions;
                    }

                    codeAttributeEditor.replaceInstruction(offset,
                                                           replacementInstructions);
                }
            }
            else
            {
                visitAnyInstruction(clazz, method, codeAttribute, offset, simpleInstruction);
            }
        }


        public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
        {
            if (instructionUsageMarker.isInstructionNecessary(offset))
            {
                if (branchInstruction.stackPopCount(clazz) > 0 &&
                    !instructionUsageMarker.isStackEntryPresentBefore(offset, instructionUsageMarker.getStackBefore(offset).size() - 1))
                {
                    // Replace the branch instruction by a simple goto.
                    Instruction replacementInstruction = new BranchInstruction(InstructionConstants.OP_GOTO,
                                                                               branchInstruction.branchOffset);
                    codeAttributeEditor.replaceInstruction(offset,
                                                           replacementInstruction);

                    if (DEBUG) System.out.println("  Replacing branch instruction "+branchInstruction.toString(offset)+" by "+replacementInstruction.toString());
                }
            }
            else
            {
                visitAnyInstruction(clazz, method, codeAttribute, offset, branchInstruction);
            }
        }


        public void visitAnySwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SwitchInstruction switchInstruction)
        {
            if (instructionUsageMarker.isInstructionNecessary(offset))
            {
                if (switchInstruction.stackPopCount(clazz) > 0 &&
                    !instructionUsageMarker.isStackEntryPresentBefore(offset, instructionUsageMarker.getStackBefore(offset).size() - 1))
                {
                    // Replace the switch instruction by a simple goto.
                    Instruction replacementInstruction = new BranchInstruction(InstructionConstants.OP_GOTO,
                                                                               switchInstruction.defaultOffset);
                    codeAttributeEditor.replaceInstruction(offset,
                                                           replacementInstruction);

                    if (DEBUG) System.out.println("  Replacing switch instruction "+switchInstruction.toString(offset)+" by "+replacementInstruction.toString());
                }
            }
            else
            {
                visitAnyInstruction(clazz, method, codeAttribute, offset, switchInstruction);
            }
        }


        /**
         * Returns whether the given instruction is a dup or swap instruction
         * (dup, dup_x1, dup_x2, dup2, dup2_x1, dup2_x2, swap).
         */
        private boolean isDupOrSwap(Instruction instruction)
        {
            return instruction.opcode >= InstructionConstants.OP_DUP &&
                   instruction.opcode <= InstructionConstants.OP_SWAP;
        }


        /**
         * Returns a dup/swap opcode that is corrected for the stack entries
         * that are present before the instruction and necessary after the
         * instruction. The returned integer opcode may contain multiple byte
         * opcodes (least significant byte first).
         * @param instructionOffset the offset of the dup/swap instruction.
         * @param dupSwapOpcode     the original dup/swap opcode.
         * @param topBefore         the index of the top stack entry before
         *                          the instruction (counting from the bottom).
         * @param topAfter          the index of the top stack entry after
         *                          the instruction (counting from the bottom).
         * @return the corrected opcode.
         */
        private int fixDupSwap(int  instructionOffset,
                               byte dupSwapOpcode,
                               int  topBefore,
                               int  topAfter)
        {
            switch (dupSwapOpcode)
            {
                case InstructionConstants.OP_DUP:     return fixedDup    (instructionOffset, topBefore, topAfter);
                case InstructionConstants.OP_DUP_X1:  return fixedDup_x1 (instructionOffset, topBefore, topAfter);
                case InstructionConstants.OP_DUP_X2:  return fixedDup_x2 (instructionOffset, topBefore, topAfter);
                case InstructionConstants.OP_DUP2:    return fixedDup2   (instructionOffset, topBefore, topAfter);
                case InstructionConstants.OP_DUP2_X1: return fixedDup2_x1(instructionOffset, topBefore, topAfter);
                case InstructionConstants.OP_DUP2_X2: return fixedDup2_x2(instructionOffset, topBefore, topAfter);
                case InstructionConstants.OP_SWAP:    return fixedSwap   (instructionOffset, topBefore, topAfter);
                default: throw new IllegalArgumentException("Not a dup/swap opcode ["+dupSwapOpcode+"]");
            }
        }


        private int fixedDup(int instructionOffset, int topBefore, int topAfter)
        {
            boolean stackEntryPresent0 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore);

            boolean stackEntryNecessary0 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter);
            boolean stackEntryNecessary1 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 1);

            // Figure out which stack entries should be moved,
            // copied, or removed.
            return
                stackEntryNecessary0 ?
                    stackEntryNecessary1 ? DUP : // ...O -> ...OO
                                           NOP : // ...O -> ...O
                stackEntryNecessary1     ? NOP : // ...O -> ...O
                stackEntryPresent0       ? POP : // ...O -> ...
                                           NOP;  // ...  -> ...
        }


        private int fixedDup_x1(int instructionOffset, int topBefore, int topAfter)
        {
            boolean stackEntryPresent0 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore);
            boolean stackEntryPresent1 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore - 1);

            boolean stackEntryNecessary0 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter);
            boolean stackEntryNecessary1 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 1);
            boolean stackEntryNecessary2 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 2);

            // Figure out which stack entries should be moved,
            // copied, or removed.
            return
                stackEntryNecessary1 ?
                    stackEntryNecessary2 ?
                        stackEntryNecessary0 ? DUP_X1       : // ...XO -> ...OXO
                                               SWAP         : // ...XO -> ...OX
                    // !stackEntryNecessary2
                        stackEntryNecessary0 ? NOP          : // ...XO -> ...XO
                        stackEntryPresent0   ? POP          : // ...XO -> ...X
                                               NOP          : // ...X  -> ...X
                stackEntryPresent1 ?
                    stackEntryNecessary2 ?
                        stackEntryNecessary0 ? SWAP_POP_DUP : // ...XO -> ...OO
                                               POP_X1       : // ...XO -> ...O
                    // !stackEntryNecessary2
                        stackEntryNecessary0 ? POP_X1       : // ...XO -> ...O
                        stackEntryPresent0   ? POP2         : // ...XO -> ...
                                               POP          : // ...X  -> ...
                // !stackEntryPresent1
                    stackEntryNecessary2 ?
                        stackEntryNecessary0 ? DUP          : // ...O -> ...OO
                                               NOP          : // ...O -> ...O
                    // !stackEntryNecessary2
                        stackEntryNecessary0 ? NOP          : // ...O -> ...O
                        stackEntryPresent0   ? POP          : // ...O -> ...
                                               NOP;           // ...  -> ...
        }


        private int fixedDup_x2(int instructionOffset, int topBefore, int topAfter)
        {
            boolean stackEntryPresent0 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore);
            boolean stackEntryPresent1 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore - 1);
            boolean stackEntryPresent2 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore - 2);

            boolean stackEntryNecessary0 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter);
            boolean stackEntryNecessary1 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 1);
            boolean stackEntryNecessary2 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 2);
            boolean stackEntryNecessary3 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 3);

            // Figure out which stack entries should be moved,
            // copied, or removed.
            return
                stackEntryNecessary1 ?
                    stackEntryNecessary2 ?
                        stackEntryNecessary3 ?
                            stackEntryNecessary0 ? DUP_X2              : // ...XYO -> ...OXYO
                                                   MOV_X2              : // ...XYO -> ...OXY
                        // !stackEntryNecessary3
                            stackEntryNecessary0 ? NOP                 : // ...XYO -> ...XYO
                            stackEntryPresent0   ? POP                 : // ...XYO -> ...XY
                                                   NOP                 : // ...XY  -> ...XY
                    stackEntryPresent2 ?
                        stackEntryNecessary3 ?
                            stackEntryNecessary0 ? DUP2_X1_POP3_DUP_X1 : // ...XYO -> ...OYO
                                                   SWAP_DUP2_X1_POP3   : // ...XYO -> ...OY
                        // !stackEntryNecessary3
                            stackEntryNecessary0 ? POP_X2              : // ...XYO -> ...YO
                            stackEntryPresent0   ? POP_SWAP_POP        : // ...XYO -> ...Y
                                                   POP_X1              : // ...XY  -> ...Y
                    // !stackEntryPresent2
                        stackEntryNecessary3 ?
                            stackEntryNecessary0 ? DUP_X1              : // ...YO -> ...OYO
                                                   SWAP                : // ...YO -> ...OY
                        // !stackEntryNecessary3
                            stackEntryNecessary0 ? NOP                 : // ...YO -> ...YO
                            stackEntryPresent0   ? POP                 : // ...YO -> ...Y
                                                   NOP                 : // ...Y  -> ...Y
                stackEntryPresent1 ?
                    stackEntryNecessary2 ?
                        stackEntryNecessary3 ?
                            stackEntryNecessary0 ? SWAP_POP_DUP_X1     : // ...XYO -> ...OXO
                                                   DUP_X2_POP2         : // ...XYO -> ...OX
                        // !stackEntryNecessary3
                            stackEntryNecessary0 ? POP_X1              : // ...XYO -> ...XO
                            stackEntryPresent0   ? POP2                : // ...XYO -> ...X
                                                   POP                 : // ...XY  -> ...X
                    stackEntryPresent2 ?
                        stackEntryNecessary3 ?
                            stackEntryNecessary0 ? DUP_X2_POP3_DUP     : // ...XYO -> ...OO
                                                   POP2_X1             : // ...XYO -> ...O
                        // !stackEntryNecessary3
                            stackEntryNecessary0 ? POP2_X1             : // ...XYO -> ...O
                            stackEntryPresent0   ? POP3                : // ...XYO -> ...
                                                   POP2                : // ...XY  -> ...
                    // !stackEntryPresent2
                        stackEntryNecessary3 ?
                            stackEntryNecessary0 ? SWAP_POP_DUP        : // ...YO -> ...OO
                                                   POP_X1              : // ...YO -> ...O
                        // !stackEntryNecessary3
                            stackEntryNecessary0 ? POP_X1              : // ...YO -> ...O
                            stackEntryPresent0   ? POP2                : // ...YO -> ...
                                                   POP                 : // ...Y  -> ...
                // !stackEntryPresent1
                    stackEntryNecessary2 ?
                        stackEntryNecessary3 ?
                            stackEntryNecessary0 ? DUP_X1              : // ...XO -> ...OXO
                                                   SWAP                : // ...XO -> ...OX
                        // !stackEntryNecessary3
                            stackEntryNecessary0 ? NOP                 : // ...XO -> ...XO
                            stackEntryPresent0   ? POP                 : // ...XO -> ...X
                                                   NOP                 : // ...X  -> ...X
                    stackEntryPresent2 ?
                        stackEntryNecessary3 ?
                            stackEntryNecessary0 ? SWAP_POP_DUP        : // ...XO -> ...OO
                                                   POP_X1              : // ...XO -> ...O
                        // !stackEntryNecessary3
                            stackEntryNecessary0 ? POP_X1              : // ...XO -> ...O
                            stackEntryPresent0   ? POP2                : // ...XO -> ...
                                                   POP                 : // ...X  -> ...
                    // !stackEntryPresent2
                        stackEntryNecessary3 ?
                            stackEntryNecessary0 ? DUP                 : // ...O -> ...OO
                                                   NOP                 : // ...O -> ...O
                        // !stackEntryNecessary3
                            stackEntryNecessary0 ? NOP                 : // ...O -> ...O
                            stackEntryPresent0   ? POP                 : // ...O -> ...
                                                   NOP;                  // ...  -> ...
        }


        private int fixedDup2(int instructionOffset, int topBefore, int topAfter)
        {
            boolean stackEntryPresent0 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore);
            boolean stackEntryPresent1 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore - 1);

            boolean stackEntryNecessary0 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter);
            boolean stackEntryNecessary1 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 1);
            boolean stackEntryNecessary2 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 2);
            boolean stackEntryNecessary3 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 3);

            // Figure out which stack entries should be moved,
            // copied, or removed.
            return
                stackEntryNecessary3 ?
                    stackEntryNecessary2 ?
                        stackEntryNecessary1 ?
                            stackEntryNecessary0 ? DUP2             : // ...AB -> ...ABAB
                                                   SWAP_DUP_X1      : // ...AB -> ...ABA
                        // !stackEntryNecessary1
                            stackEntryNecessary0 ? DUP              : // ...AB -> ...ABB
                                                   NOP              : // ...AB -> ...AB
                    // !stackEntryNecessary2
                        stackEntryNecessary1 ?
                            stackEntryNecessary0 ? SWAP_DUP_X1_SWAP : // ...AB -> ...AAB
                            stackEntryPresent0   ? POP_DUP          : // ...AB -> ...AA
                                                   DUP              : // ...A  -> ...AA
                        // !stackEntryNecessary1
                            stackEntryNecessary0 ? NOP              : // ...AB -> ...AB
                            stackEntryPresent0   ? POP              : // ...AB -> ...A
                                                   NOP              : // ...A  -> ...A
                // !stackEntryNecessary3
                    stackEntryNecessary2 ?
                        stackEntryNecessary1 ?
                            stackEntryNecessary0 ? DUP_X1           : // ...AB -> ...BAB
                                                   SWAP             : // ...AB -> ...BA
                        stackEntryPresent1 ?
                            stackEntryNecessary0 ? SWAP_POP_DUP     : // ...AB -> ...BB
                                                   POP_X1           : // ...AB -> ...B
                        // !stackEntryPresent1
                            stackEntryNecessary0 ? POP              : // ...B  -> ...BB
                                                   NOP              : // ...B  -> ...B
                    // !stackEntryNecessary2
                        stackEntryNecessary1 ?
                            stackEntryNecessary0 ? NOP              : // ...AB -> ...AB
                            stackEntryPresent0   ? POP              : // ...AB -> ...A
                                                   NOP              : // ...A  -> ...A
                        stackEntryPresent1 ?
                            stackEntryNecessary0 ? POP_X1           : // ...AB -> ...B
                            stackEntryPresent0   ? POP2             : // ...AB -> ...
                                                   POP              : // ...A  -> ...
                        // !stackEntryPresent1
                            stackEntryNecessary0 ? NOP              : // ...B  -> ...B
                            stackEntryPresent0   ? POP              : // ...B  -> ...
                                                   NOP;               // ...   -> ...
        }


        private int fixedDup2_x1(int instructionOffset, int topBefore, int topAfter)
        {
            boolean stackEntryPresent0 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore);
            boolean stackEntryPresent1 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore - 1);
            boolean stackEntryPresent2 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore - 2);

            boolean stackEntryNecessary0 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter);
            boolean stackEntryNecessary1 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 1);
            boolean stackEntryNecessary2 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 2);
            boolean stackEntryNecessary3 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 3);
            boolean stackEntryNecessary4 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 4);

            // Figure out which stack entries should be moved,
            // copied, or removed.
            return
                stackEntryNecessary4 ?
                    stackEntryNecessary3 ?
                        stackEntryNecessary2 ?
                            stackEntryNecessary1 ?
                                stackEntryNecessary0 ? DUP2_X1             : // ...XAB -> ...ABXAB
                                                       DUP2_X1_POP         : // ...XAB -> ...ABXA
                            // !stackEntryNecessary1
                                stackEntryNecessary0 ? UNSUPPORTED         : // ...XAB -> ...ABXB
                                                       MOV2_X1             : // ...XAB -> ...ABX
                        stackEntryPresent2   ?
                            stackEntryNecessary1 ?
                                stackEntryNecessary0 ? DUP2_X1_POP3_DUP2   : // ...XAB -> ...ABAB
                                                       UNSUPPORTED         : // ...XAB -> ...ABA
                            // !stackEntryNecessary1
                                stackEntryNecessary0 ? DUP2_X1_POP3_DUP    : // ...XAB -> ...ABB
                                                       POP_X2              : // ...XAB -> ...AB
                        // !stackEntryNecessary2
                            stackEntryNecessary1 ?
                                stackEntryNecessary0 ? DUP2                : // ...AB  -> ...ABAB
                                                       SWAP_DUP_X1         : // ...AB  -> ...ABA
                            // !stackEntryNecessary1
                                stackEntryNecessary0 ? DUP                 : // ...AB  -> ...ABB
                                                       NOP                 : // ...AB  -> ...AB
                    // !stackEntryNecessary3
                        stackEntryNecessary2 ?
                            stackEntryNecessary1 ?
                                stackEntryNecessary0 ? UNSUPPORTED         : // ...XAB -> ...AXAB
                                stackEntryPresent0   ? POP_DUP_X1          : // ...XAB -> ...AXA
                                                       DUP_X1              : // ...XA  -> ...AXA
                            // !stackEntryNecessary1
                                stackEntryNecessary0 ? UNSUPPORTED         : // ...XAB -> ...AXB
                                stackEntryPresent0   ? POP_SWAP            : // ...XAB -> ...AX
                                                       SWAP                : // ...XA  -> ...AX
                        stackEntryPresent2   ?
                            stackEntryNecessary1 ?
                                stackEntryNecessary0 ? UNSUPPORTED         : // ...XAB -> ...AAB
                                stackEntryPresent0   ? POP_SWAP_POP_DUP    : // ...XAB -> ...AA
                                                       SWAP_POP_DUP        : // ...XA  -> ...AA
                            // !stackEntryNecessary1
                                stackEntryNecessary0 ? POP_X2              : // ...XAB -> ...AB
                                stackEntryPresent0   ? POP_SWAP_POP        : // ...XAB -> ...A
                                                       POP_X1              : // ...XA  -> ...A
                        // !stackEntryNecessary2
                            stackEntryNecessary1 ?
                                stackEntryNecessary0 ? SWAP_DUP_X1_SWAP    : // ...AB  -> ...AAB
                                stackEntryPresent0   ? POP_DUP             : // ...AB  -> ...AA
                                                       DUP                 : // ...A   -> ...AA
                            // !stackEntryNecessary1
                                stackEntryNecessary0 ? NOP                 : // ...AB  -> ...AB
                                stackEntryPresent0   ? POP                 : // ...AB  -> ...A
                                                       NOP                 : // ...A   -> ...A
                // !stackEntryNecessary4
                    stackEntryNecessary3 ?
                        stackEntryNecessary2 ?
                            stackEntryNecessary1 ?
                                stackEntryNecessary0 ? DUP_X2              : // ...XAB -> ...BXAB
                                                       DUP_X2_POP          : // ...XAB -> ...BXA
                            stackEntryPresent1   ?
                                stackEntryNecessary0 ? SWAP_POP_DUP_X1     : // ...XAB -> ...BXB
                                                       DUP_X2_POP2         : // ...XAB -> ...BX
                            // !stackEntryNecessary1
                                stackEntryNecessary0 ? POP_X2              : // ...XB  -> ...BXB
                                                       SWAP                : // ...XB  -> ...BX
                        stackEntryPresent2   ?
                            stackEntryNecessary1 ?
                                stackEntryNecessary0 ? DUP2_X1_POP3_DUP_X1 : // ...XAB -> ...BAB
                                                       SWAP_DUP_X1_POP3    : // ...XAB -> ...BA
                            stackEntryPresent1   ?
                                stackEntryNecessary0 ? DUP_X2_POP3_DUP     : // ...XAB -> ...BB
                                                       POP2_X1             : // ...XAB -> ...B
                            // !stackEntryNecessary1
                                stackEntryNecessary0 ? SWAP_POP_DUP        : // ...XB  -> ...BB
                                                       POP_X1              : // ...XB  -> ...B
                        // !stackEntryNecessary2
                            stackEntryNecessary1 ?
                                stackEntryNecessary0 ? DUP_X1              : // ...AB  -> ...BAB
                                                       SWAP                : // ...AB  -> ...BA
                            stackEntryPresent1   ?
                                stackEntryNecessary0 ? SWAP_POP_DUP        : // ...AB  -> ...BB
                                                       POP_X1              : // ...AB  -> ...B
                            // !stackEntryNecessary1
                                stackEntryNecessary0 ? DUP                 : // ...B   -> ...BB
                                                       NOP                 : // ...B   -> ...B
                    // !stackEntryNecessary3
                        stackEntryNecessary2 ?
                            stackEntryNecessary1 ?
                                stackEntryNecessary0 ? NOP                 : // ...XAB -> ...XAB
                                stackEntryPresent0   ? POP                 : // ...XAB -> ...XA
                                                       NOP                 : // ...XA  -> ...XA
                            stackEntryPresent1   ?
                                stackEntryNecessary0 ? POP_X1              : // ...XAB -> ...XB
                                stackEntryPresent0   ? POP2                : // ...XAB -> ...X
                                                       POP                 : // ...XA  -> ...X
                            // !stackEntryNecessary1
                                stackEntryNecessary0 ? NOP                 : // ...XB  -> ...XB
                                stackEntryPresent0   ? POP                 : // ...XB  -> ...X
                                                       NOP                 : // ...X   -> ...X
                        stackEntryPresent2   ?
                            stackEntryNecessary1 ?
                                stackEntryNecessary0 ? POP_X2              : // ...XAB -> ...AB
                                stackEntryPresent0   ? POP_SWAP_POP        : // ...XAB -> ...A
                                                       POP_X1              : // ...XA  -> ...A
                            stackEntryPresent1   ?
                                stackEntryNecessary0 ? POP2_X1             : // ...XAB -> ...B
                                stackEntryPresent0   ? POP3                : // ...XAB -> ...
                                                       POP2                : // ...XA  -> ...
                            // !stackEntryNecessary1
                                stackEntryNecessary0 ? POP_X1              : // ...XB  -> ...B
                                stackEntryPresent0   ? POP2                : // ...XB  -> ...
                                                       POP                 : // ...X   -> ...
                        // !stackEntryNecessary2
                            stackEntryNecessary1 ?
                                stackEntryNecessary0 ? NOP                 : // ...AB  -> ...AB
                                stackEntryPresent0   ? POP                 : // ...AB  -> ...A
                                                       NOP                 : // ...A   -> ...A
                            stackEntryPresent1   ?
                                stackEntryNecessary0 ? POP_X1              : // ...AB  -> ...B
                                stackEntryPresent0   ? POP2                : // ...AB  -> ...
                                                       POP                 : // ...A   -> ...
                            // !stackEntryNecessary1
                                stackEntryNecessary0 ? NOP                 : // ...B   -> ...B
                                stackEntryPresent0   ? POP                 : // ...B   -> ...
                                                       NOP;                  // ...    -> ...
        }


        private int fixedDup2_x2(int instructionOffset, int topBefore, int topAfter)
        {
            // We're currently assuming the value to be duplicated
            // is a long or a double, taking up two slots, or at
            // least consistent.
            boolean stackEntriesPresent01 = instructionUsageMarker.isStackEntriesPresentBefore(instructionOffset, topBefore, topBefore - 1);
            boolean stackEntryPresent2    = instructionUsageMarker.isStackEntryPresentBefore(  instructionOffset, topBefore - 2);
            boolean stackEntryPresent3    = instructionUsageMarker.isStackEntryPresentBefore(  instructionOffset, topBefore - 3);

            boolean stackEntriesNecessary01 = instructionUsageMarker.isStackEntriesNecessaryAfter(instructionOffset, topAfter, topAfter - 1);
            boolean stackEntryNecessary2    = instructionUsageMarker.isStackEntryNecessaryAfter(  instructionOffset, topAfter - 2);
            boolean stackEntryNecessary3    = instructionUsageMarker.isStackEntryNecessaryAfter(  instructionOffset, topAfter - 3);
            boolean stackEntriesNecessary45 = instructionUsageMarker.isStackEntriesNecessaryAfter(instructionOffset, topAfter - 4, topAfter - 5);

            // Figure out which stack entries should be moved,
            // copied, or removed.
            return
                stackEntryNecessary2 ?
                    stackEntryNecessary3 ?
                        stackEntriesNecessary45 ?
                            stackEntriesNecessary01 ? DUP2_X2           : // ...XYAB -> ...ABXYAB
                                                      MOV2_X2           : // ...XYAB -> ...ABXY
                        // !stackEntriesNecessary45
                            stackEntriesNecessary01 ? NOP               : // ...XYAB -> ...XYAB
                            stackEntriesPresent01   ? POP2              : // ...XYAB -> ...XY
                                                      NOP               : // ...XY   -> ...XY
                    stackEntryPresent3 ?
                        stackEntriesNecessary45 ?
                            stackEntriesNecessary01 ? UNSUPPORTED       : // ...XYAB -> ...ABYAB
                                                      DUP2_X2_SWAP_POP  : // ...XYAB -> ...ABY
                        // !stackEntriesNecessary45
                            stackEntriesNecessary01 ? POP_X3            : // ...XYAB -> ...YAB
                            stackEntriesPresent01   ? POP2_SWAP_POP     : // ...XYAB -> ...Y
                                                      POP_X1            : // ...XY   -> ...Y
                    // !stackEntryPresent3
                        stackEntriesNecessary45 ?
                            stackEntriesNecessary01 ? DUP2_X1           : // ...YAB -> ...ABYAB
                                                      MOV2_X1           : // ...YAB -> ...ABY
                        // !stackEntriesNecessary45
                            stackEntriesNecessary01 ? NOP               : // ...YAB -> ...YAB
                            stackEntriesPresent01   ? POP2              : // ...YAB -> ...Y
                                                      NOP               : // ...Y   -> ...Y
                stackEntryPresent2 ?
                    stackEntryNecessary3 ?
                        stackEntriesNecessary45 ?
                            stackEntriesNecessary01 ? UNSUPPORTED       : // ...XYAB -> ...ABXAB
                                                      DUP2_X2_POP3      : // ...XYAB -> ...ABX
                        // !stackEntriesNecessary45
                            stackEntriesNecessary01 ? POP_X2            : // ...XYAB -> ...XAB
                            stackEntriesPresent01   ? POP3              : // ...XYAB -> ...X
                                                      POP               : // ...XY   -> ...X
                    stackEntryPresent3 ?
                        stackEntriesNecessary45 ?
                            stackEntriesNecessary01 ? UNSUPPORTED       : // ...XYAB -> ...ABAB
                                                      POP2_X2           : // ...XYAB -> ...AB
                        // !stackEntriesNecessary45
                            stackEntriesNecessary01 ? POP2_X2           : // ...XYAB -> ...AB
                            stackEntriesPresent01   ? POP4              : // ...XYAB -> ...
                                                      POP2              : // ...XY   -> ...
                    // !stackEntryPresent3
                        stackEntriesNecessary45 ?
                            stackEntriesNecessary01 ? DUP2_X1_POP3_DUP2 : // ...YAB -> ...ABAB
                                                      POP_X2            : // ...YAB -> ...AB
                        // !stackEntriesNecessary45
                            stackEntriesNecessary01 ? POP_X2            : // ...YAB -> ...AB
                            stackEntriesPresent01   ? POP3              : // ...YAB -> ...
                                                      POP               : // ...Y   -> ...
                // !stackEntryPresent2
                    stackEntryNecessary3 ?
                        stackEntriesNecessary45 ?
                            stackEntriesNecessary01 ? DUP2_X1           : // ...XAB -> ...ABXAB
                                                      MOV2_X1           : // ...XAB -> ...ABX
                        // !stackEntriesNecessary45
                            stackEntriesNecessary01 ? NOP               : // ...XAB -> ...XAB
                            stackEntriesPresent01   ? POP2              : // ...XAB -> ...X
                                                      NOP               : // ...X   -> ...X
                    stackEntryPresent3 ?
                        stackEntriesNecessary45 ?
                            stackEntriesNecessary01 ? DUP2_X1_POP3_DUP2 : // ...XAB -> ...ABAB
                                                      POP_X2            : // ...XAB -> ...AB
                        // !stackEntriesNecessary45
                            stackEntriesNecessary01 ? POP_X2            : // ...XAB -> ...AB
                            stackEntriesPresent01   ? POP3              : // ...XAB -> ...
                                                      POP               : // ...X   -> ...
                    // !stackEntryPresent3
                        stackEntriesNecessary45 ?
                            stackEntriesNecessary01 ? DUP2              : // ...AB -> ...ABAB
                                                      NOP               : // ...AB -> ...AB
                        // !stackEntriesNecessary45
                            stackEntriesNecessary01 ? NOP               : // ...AB -> ...AB
                            stackEntriesPresent01   ? POP2              : // ...AB -> ...
                                                      NOP;               // ...   -> ...
        }


        private int fixedSwap(int instructionOffset, int topBefore, int topAfter)
        {
            boolean stackEntryPresent0 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore);
            boolean stackEntryPresent1 = instructionUsageMarker.isStackEntryPresentBefore(instructionOffset, topBefore - 1);

            boolean stackEntryNecessary0 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter);
            boolean stackEntryNecessary1 = instructionUsageMarker.isStackEntryNecessaryAfter(instructionOffset, topAfter - 1);

            // Figure out which stack entries should be moved
            // or removed.
            return
                stackEntryNecessary0 ?
                    stackEntryNecessary1 ? SWAP   : // ...AB -> ...BA
                    stackEntryPresent0   ? POP    : // ...AB -> ...A
                                           NOP    : // ...A  -> ...A
                stackEntryPresent1       ? POP_X1 : // ...AB -> ...B
                                           NOP;     // ...B -> ...B
        }
    }


    /**
     * This InstructionVisitor deletes all visited instructions.
     */
    private class MyInstructionDeleter
    extends       SimplifiedVisitor
    implements    InstructionVisitor
    {
        // Implementations for InstructionVisitor.

        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
        {
            codeAttributeEditor.deleteInstruction(offset);

            // We're allowing edits on deleted instructions.
            //codeAttributeEditor.insertBeforeInstruction(offset, (Instruction)null);
            //codeAttributeEditor.replaceInstruction(offset,      (Instruction)null);
            //codeAttributeEditor.insertAfterInstruction(offset,  (Instruction)null);

            // Visit the instruction, if required.
            if (extraDeletedInstructionVisitor != null)
            {
                instruction.accept(clazz, method, codeAttribute, offset, extraDeletedInstructionVisitor);
            }
        }
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        // Is the catch handler necessary?
        if (!instructionUsageMarker.isTraced(exceptionInfo.u2handlerPC))
        {
            // Make the code block empty, so the code editor can remove it.
            exceptionInfo.u2endPC = exceptionInfo.u2startPC;
        }
    }


    // Small utility methods.

    /**
     * Returns whether any traced but unnecessary instruction between the two
     * given offsets is branching over the second given offset.
     */
    private boolean isAnyUnnecessaryInstructionBranchingOver(int instructionOffset1,
                                                             int instructionOffset2)
    {
        for (int offset = instructionOffset1; offset < instructionOffset2; offset++)
        {
            // Is it a traced but unmarked straddling branch?
            if (instructionUsageMarker.isTraced(offset) &&
                !instructionUsageMarker.isInstructionNecessary(offset)   &&
                isAnyLargerThan(instructionUsageMarker.branchTargets(offset),
                                instructionOffset2))
            {
                return true;
            }
        }

        return false;
    }


    /**
     * Returns whether any of the given instruction offsets (at least one)
     * is larger than the given offset.
     */
    private boolean isAnyLargerThan(InstructionOffsetValue instructionOffsets,
                                    int                    instructionOffset)
    {
        if (instructionOffsets != null)
        {
            // Loop over all instruction offsets.
            int branchCount = instructionOffsets.instructionOffsetCount();
            if (branchCount > 0)
            {
                for (int branchIndex = 0; branchIndex < branchCount; branchIndex++)
                {
                    // Is the offset larger than the reference offset?
                    if (instructionOffsets.instructionOffset(branchIndex) > instructionOffset)
                    {
                        return true;
                    }
                }
            }
        }

        return false;
    }


    /**
     * Pushes a specified type of stack entry before or at the given offset.
     * The instruction is marked as necessary.
     */
    private void insertPushInstructions(int     offset,
                                        boolean replace,
                                        boolean before,
                                        int     computationalType)
    {
        // We can edit an instruction without marking it.
        //markInstruction(offset);

        // Create a simple push instrucion.
        Instruction replacementInstruction =
            new SimpleInstruction(pushOpcode(computationalType));

        if (DEBUG) System.out.println(": "+replacementInstruction.toString(offset));

        // Replace or insert the push instruction.
        insertInstruction(offset, replace, before, replacementInstruction);
    }


    /**
     * Returns the opcode of a push instruction corresponding to the given
     * computational type.
     * @param computationalType the computational type to be pushed on the stack.
     */
    private byte pushOpcode(int computationalType)
    {
        switch (computationalType)
        {
            case Value.TYPE_INTEGER:            return InstructionConstants.OP_ICONST_0;
            case Value.TYPE_LONG:               return InstructionConstants.OP_LCONST_0;
            case Value.TYPE_FLOAT:              return InstructionConstants.OP_FCONST_0;
            case Value.TYPE_DOUBLE:             return InstructionConstants.OP_DCONST_0;
            case Value.TYPE_REFERENCE:
            case Value.TYPE_INSTRUCTION_OFFSET: return InstructionConstants.OP_ACONST_NULL;
        }

        throw new IllegalArgumentException("No push opcode for computational type ["+computationalType+"]");
    }


    /**
     * Pops the given number of stack entries at or after the given offset.
     * The instructions are marked as necessary.
     */
    private void insertPopInstructions(int     offset,
                                       boolean replace,
                                       boolean before,
                                       int     popCount)
    {
        // We can edit an instruction without marking it.
        //markInstruction(offset);

        switch (popCount)
        {
            case 1:
            {
                // Replace or insert a single pop instruction.
                Instruction popInstruction =
                    new SimpleInstruction(InstructionConstants.OP_POP);

                insertInstruction(offset, replace, before, popInstruction);
                break;
            }
            case 2:
            {
                // Replace or insert a single pop2 instruction.
                Instruction popInstruction =
                    new SimpleInstruction(InstructionConstants.OP_POP2);

                insertInstruction(offset, replace, before, popInstruction);
                break;
            }
            default:
            {
                // Replace or insert the specified number of pop instructions.
                Instruction[] popInstructions =
                    new Instruction[popCount / 2 + popCount % 2];

                Instruction popInstruction =
                    new SimpleInstruction(InstructionConstants.OP_POP2);

                for (int index = 0; index < popCount / 2; index++)
                {
                      popInstructions[index] = popInstruction;
                }

                if (popCount % 2 == 1)
                {
                    popInstruction =
                        new SimpleInstruction(InstructionConstants.OP_POP);

                    popInstructions[popCount / 2] = popInstruction;
                }

                insertInstructions(offset,
                                   replace,
                                   before,
                                   popInstruction,
                                   popInstructions);
                break;
            }
        }
    }


    /**
     * Inserts or replaces the given instruction at the given offset.
     */
    private void insertInstruction(int         offset,
                                   boolean     replace,
                                   boolean     before,
                                   Instruction instruction)
    {
        if (replace)
        {
            codeAttributeEditor.replaceInstruction(offset, instruction);

            if (extraAddedInstructionVisitor != null &&
                !instructionUsageMarker.isInstructionNecessary(offset))
            {
                instruction.accept(null, null, null, offset, extraAddedInstructionVisitor);
            }
        }
        else
        {
            if (before)
            {
                codeAttributeEditor.insertBeforeInstruction(offset, instruction);
            }
            else
            {
                codeAttributeEditor.insertAfterInstruction(offset, instruction);
            }

            if (extraAddedInstructionVisitor != null)
            {
                instruction.accept(null, null, null, offset, extraAddedInstructionVisitor);
            }
        }
    }


    /**
     * Inserts or replaces the given instruction at the given offset.
     */
    private void insertInstructions(int           offset,
                                    boolean       replace,
                                    boolean       before,
                                    Instruction   instruction,
                                    Instruction[] instructions)
    {
        if (replace)
        {
            codeAttributeEditor.replaceInstruction(offset, instructions);

            if (extraAddedInstructionVisitor != null)
            {
                if (!instructionUsageMarker.isInstructionNecessary(offset))
                {
                    instruction.accept(null, null, null, offset, extraAddedInstructionVisitor);
                }

                for (int index = 1; index < instructions.length; index++)
                {
                    instructions[index].accept(null, null, null, offset, extraAddedInstructionVisitor);
                }
            }
        }
        else
        {
            if (before)
            {
                codeAttributeEditor.insertBeforeInstruction(offset, instructions);
            }
            else
            {
                codeAttributeEditor.insertAfterInstruction(offset, instructions);
            }

            for (int index = 0; index < instructions.length; index++)
            {
                if (extraAddedInstructionVisitor != null)
                {
                    instructions[index].accept(null, null, null, offset, extraAddedInstructionVisitor);
                }
            }
        }
    }


    /**
     * Replaces the instruction at a given offset by a static invocation.
     */
    private void replaceByStaticInvocation(Clazz               clazz,
                                           int                 offset,
                                           ConstantInstruction constantInstruction)
    {
        // Remember the replacement instruction.
        Instruction replacementInstruction =
             new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC,
                                     constantInstruction.constantIndex);

        if (DEBUG) System.out.println("  Replacing by static invocation "+constantInstruction.toString(offset)+" -> "+replacementInstruction.toString());

        codeAttributeEditor.replaceInstruction(offset, replacementInstruction);
    }


    /**
     * Replaces the given instruction by an infinite loop.
     */
    private void replaceByInfiniteLoop(Clazz clazz,
                                       int   offset)
    {
        if (DEBUG) System.out.println("  Inserting infinite loop at ["+offset+"]");

        // We can edit an instruction without marking it.
        //markInstruction(offset);

        // Replace the instruction by an infinite loop.
        Instruction replacementInstruction =
            new BranchInstruction(InstructionConstants.OP_GOTO, 0);

        codeAttributeEditor.replaceInstruction(offset, replacementInstruction);
    }
}