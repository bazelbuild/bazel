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
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.evaluation.TracedStack;
import proguard.evaluation.value.*;
import proguard.optimize.info.*;
import proguard.util.ArrayUtil;

import java.util.*;

/**
 * This AttributeVisitor marks necessary instructions in the code attributes
 * that it visits, based on partial evaluation.
 *
 * @see NoSideEffectClassMarker
 * @see SideEffectClassMarker
 * @see ReadWriteFieldMarker
 * @see NoSideEffectMethodMarker
 * @see NoExternalSideEffectMethodMarker
 * @see SideEffectMethodMarker
 * @see ParameterEscapeMarker
 *
 * @author Eric Lafortune
 */
public class InstructionUsageMarker
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    //*
    private static final boolean DEBUG          = false;
    private static final boolean DEBUG_RESULTS  = false;
    /*/
    private static boolean DEBUG          = System.getProperty("ium") != null;
    private static boolean DEBUG_RESULTS  = DEBUG;
    //*/

    private final PartialEvaluator                partialEvaluator;
    private final boolean                         runPartialEvaluator;
    private final PartialEvaluator                simplePartialEvaluator        = new PartialEvaluator(new TypedReferenceValueFactory());
    private final SideEffectInstructionChecker    sideEffectInstructionChecker  = new SideEffectInstructionChecker(true, true);
    private final MyParameterUsageMarker          parameterUsageMarker          = new MyParameterUsageMarker();
    private final MyInitialUsageMarker            initialUsageMarker            = new MyInitialUsageMarker();
    private final MyProducerMarker                producerMarker                = new MyProducerMarker();
    private final MyVariableInitializationMarker  variableInitializationMarker  = new MyVariableInitializationMarker();
    private final MyStackConsistencyMarker        stackConsistencyMarker        = new MyStackConsistencyMarker();
    private final MyExtraPushPopInstructionMarker extraPushPopInstructionMarker = new MyExtraPushPopInstructionMarker();

    private InstructionOffsetValue[] reverseDependencies = new InstructionOffsetValue[ClassConstants.TYPICAL_CODE_LENGTH];

    private boolean[][] stacksNecessaryAfter              = new boolean[ClassConstants.TYPICAL_CODE_LENGTH][ClassConstants.TYPICAL_STACK_SIZE];
    private boolean[][] stacksUnwantedBefore              = new boolean[ClassConstants.TYPICAL_CODE_LENGTH][ClassConstants.TYPICAL_STACK_SIZE];
    private boolean[]   instructionsNecessary             = new boolean[ClassConstants.TYPICAL_CODE_LENGTH];
    private boolean[]   extraPushPopInstructionsNecessary = new boolean[ClassConstants.TYPICAL_CODE_LENGTH];

    private int maxMarkedOffset;


    /**
     * Creates a new InstructionUsageMarker.
     */
    public InstructionUsageMarker()
    {
        this(new PartialEvaluator(), true);
    }


    /**
     * Creates a new InstructionUsageMarker.
     * @param partialEvaluator    the evaluator to be used for the analysis.
     * @param runPartialEvaluator specifies whether to run this evaluator on
     *                            every code attribute that is visited.
     */
    public InstructionUsageMarker(PartialEvaluator partialEvaluator,
                                  boolean          runPartialEvaluator)
    {
        this.partialEvaluator    = partialEvaluator;
        this.runPartialEvaluator = runPartialEvaluator;
    }


    /**
     * Returns whether the specified instruction was traced in the most
     * recently analyzed code attribute.
     */
    public boolean isTraced(int instructionOffset)
    {
        return partialEvaluator.isTraced(instructionOffset);
    }


    /**
     * Returns a filtering version of the given instruction visitor that only
     * visits traced instructions.
     */
    public InstructionVisitor tracedInstructionFilter(InstructionVisitor instructionVisitor)
    {
        return partialEvaluator.tracedInstructionFilter(instructionVisitor);
    }


    /**
     * Returns a filtering version of the given instruction visitor that only
     * visits traced or untraced instructions.
     */
    public InstructionVisitor tracedInstructionFilter(boolean            traced,
                                                      InstructionVisitor instructionVisitor)
    {
        return partialEvaluator.tracedInstructionFilter(traced, instructionVisitor);
    }


    /**
     * Returns whether the specified instruction is necessary in the most
     * recently analyzed code attribute.
     */
    public boolean isInstructionNecessary(int instructionOffset)
    {
        return instructionsNecessary[instructionOffset];
    }


    /**
     * Returns whether an extra push/pop instruction is required at the given
     * offset in the most recently analyzed code attribute.
     */
    public boolean isExtraPushPopInstructionNecessary(int instructionOffset)
    {
        return extraPushPopInstructionsNecessary[instructionOffset];
    }


    /**
     * Returns a filtering version of the given instruction visitor that only
     * visits necessary instructions.
     */
    public InstructionVisitor necessaryInstructionFilter(InstructionVisitor instructionVisitor)
    {
        return necessaryInstructionFilter(true, instructionVisitor);
    }


    /**
     * Returns a filtering version of the given instruction visitor that only
     * visits necessary or unnecessary instructions.
     */
    public InstructionVisitor necessaryInstructionFilter(boolean            necessary,
                                                         InstructionVisitor instructionVisitor)
    {
        return new MyNecessaryInstructionFilter(necessary, instructionVisitor);
    }


    /**
     * Returns the stack before execution of the instruction at the given
     * offset.
     */
    public TracedStack getStackBefore(int instructionOffset)
    {
        return partialEvaluator.getStackBefore(instructionOffset);
    }


    /**
     * Returns the stack after execution of the instruction at the given
     * offset.
     */
    public TracedStack getStackAfter(int instructionOffset)
    {
        return partialEvaluator.getStackAfter(instructionOffset);
    }


    /**
     * Returns whether the specified stack entry before the given offset is
     * unwanted, e.g. because it was intended as a method parameter that has
     * been removed.
     */
    public boolean isStackEntryUnwantedBefore(int instructionOffset,
                                              int stackIndex)
    {
        return stacksUnwantedBefore[instructionOffset][stackIndex];
    }


    /**
     * Returns whether the stack specified entries before the given offset are
     * present.
     */
    public boolean isStackEntriesPresentBefore(int instructionOffset,
                                               int stackIndex1,
                                               int stackIndex2)
    {
        boolean present1 = isStackEntryPresentBefore(instructionOffset, stackIndex1);
        boolean present2 = isStackEntryPresentBefore(instructionOffset, stackIndex2);

        //if (present1 ^ present2)
        //{
        //    throw new UnsupportedOperationException("Can't handle partial use of dup2 instructions");
        //}

        return present1 || present2;
    }


    /**
     * Returns whether the specified stack entry before the given offset is
     * present.
     * @param instructionOffset the offset of the stack entry to be checked.
     * @param stackIndex        the index of the stack entry to be checked
     *                          (counting from the bottom).
     */
    public boolean isStackEntryPresentBefore(int instructionOffset,
                                             int stackIndex)
    {
        TracedStack tracedStack =
            partialEvaluator.getStackBefore(instructionOffset);

        InstructionOffsetValue producerOffsets =
            tracedStack.getBottomProducerValue(stackIndex).instructionOffsetValue();

        return isAnyStackEntryNecessaryAfter(producerOffsets, stackIndex);
    }


    /**
     * Returns whether the stack specified entries after the given offset are
     * necessary.
     */
    public boolean isStackEntriesNecessaryAfter(int instructionOffset,
                                                int stackIndex1,
                                                int stackIndex2)
    {
        boolean present1 = isStackEntryNecessaryAfter(instructionOffset, stackIndex1);
        boolean present2 = isStackEntryNecessaryAfter(instructionOffset, stackIndex2);

        //if (present1 ^ present2)
        //{
        //    throw new UnsupportedOperationException("Can't handle partial use of dup2 instructions");
        //}

        return present1 || present2;
    }


    /**
     * Returns whether any of the stack entries after the given offsets are
     * necessary.
     * @param instructionOffsets the offsets of the stack entries to be checked.
     * @param stackIndex         the index of the stack entries to be checked
     *                           (counting from the bottom).
     */
    public boolean isAnyStackEntryNecessaryAfter(InstructionOffsetValue instructionOffsets,
                                                 int                    stackIndex)
    {
        int offsetCount = instructionOffsets.instructionOffsetCount();

        for (int offsetIndex = 0; offsetIndex < offsetCount; offsetIndex++)
        {
            if (instructionOffsets.isExceptionHandler(offsetIndex) ||
                isStackEntryNecessaryAfter(instructionOffsets.instructionOffset(offsetIndex), stackIndex))
            {
                return true;
            }
        }

        return false;
    }


    /**
     * Returns whether the specified stack entry after the given offset is
     * necessary.
     * @param instructionOffset the offset of the stack entry to be checked.
     * @param stackIndex        the index of the stack entry to be checked
     *                          (counting from the bottom).
     */
    public boolean isStackEntryNecessaryAfter(int instructionOffset,
                                              int stackIndex)
    {
        return
            (instructionOffset & InstructionOffsetValue.EXCEPTION_HANDLER) != 0 ||
            stacksNecessaryAfter[instructionOffset][stackIndex];
    }


    /**
     * Returns the instruction offsets to which the given instruction offset
     * branches in the most recently analyzed code attribute.
     */
    public InstructionOffsetValue branchTargets(int instructionOffset)
    {
        return partialEvaluator.branchTargets(instructionOffset);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
//        DEBUG = DEBUG_RESULTS =
//            clazz.getName().equals("abc/Def") &&
//            method.getName(clazz).equals("abc");

        // TODO: Remove this when the instruction usage marker has stabilized.
        // Catch any unexpected exceptions from the actual visiting method.
        try
        {
            // Process the code.
            visitCodeAttribute0(clazz, method, codeAttribute);
        }
        catch (RuntimeException ex)
        {
            System.err.println("Unexpected error while marking instruction usage after partial evaluation:");
            System.err.println("  Class       = ["+clazz.getName()+"]");
            System.err.println("  Method      = ["+method.getName(clazz)+method.getDescriptor(clazz)+"]");
            System.err.println("  Exception   = ["+ex.getClass().getName()+"] ("+ex.getMessage()+")");

            if (DEBUG)
            {
                method.accept(clazz, new ClassPrinter());
            }

            throw ex;
        }
    }


    public void visitCodeAttribute0(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        if (DEBUG_RESULTS)
        {
            System.out.println();
            System.out.println("InstructionUsageMarker ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]");
        }

        // Initialize the necessary arrays.
        initializeNecessary(codeAttribute);

        // Evaluate the method.
        if (runPartialEvaluator)
        {
            partialEvaluator.visitCodeAttribute(clazz, method, codeAttribute);
        }

        // Evaluate the method the way the JVM verifier would do it.
        simplePartialEvaluator.visitCodeAttribute(clazz, method, codeAttribute);

        int codeLength = codeAttribute.u4codeLength;

        maxMarkedOffset = -1;

        // Mark any unused method parameters on the stack.
        if (DEBUG) System.out.println("Invocation simplification:");

        codeAttribute.instructionsAccept(clazz, method,
            partialEvaluator.tracedInstructionFilter(parameterUsageMarker));


        // Mark all essential instructions that have been encountered as used.
        // Also mark infinite loops and instructions that can have side effects.
        if (DEBUG) System.out.println("Usage initialization: ");

        codeAttribute.instructionsAccept(clazz, method,
            partialEvaluator.tracedInstructionFilter(initialUsageMarker));

        if (DEBUG) System.out.println();


        // Globally mark instructions and their produced variables and stack
        // entries on which necessary instructions depend.
        // Instead of doing this recursively, we loop across all instructions,
        // starting at the highest previously unmarked instruction that has
        // been been marked.
        if (DEBUG) System.out.println("Usage marking:");

        while (maxMarkedOffset >= 0)
        {
            int offset = maxMarkedOffset;

            maxMarkedOffset = offset - 1;

            if (partialEvaluator.isTraced(offset))
            {
                if (isInstructionNecessary(offset))
                {
                    // Mark the stack/variable producers of this instruction/
                    Instruction instruction = InstructionFactory.create(codeAttribute.code,
                                                                        offset);

                    instruction.accept(clazz, method, codeAttribute, offset, producerMarker);

                    // Also mark any reverse dependencies.
                    markReverseDependencies(offset);
                }

                // Check if this instruction is a branch origin from a branch
                // that straddles some marked code.
                markStraddlingBranches(offset,
                                       partialEvaluator.branchTargets(offset),
                                       true);

                // Check if this instruction is a branch target from a branch
                // that straddles some marked code.
                markStraddlingBranches(offset,
                                       partialEvaluator.branchOrigins(offset),
                                       false);
            }

            if (DEBUG)
            {
                if (maxMarkedOffset > offset)
                {
                    System.out.println(" -> "+maxMarkedOffset);
                }
            }
        }
        if (DEBUG) System.out.println();


        // Mark variable initializations, even if  they aren't strictly necessary.
        // The virtual machine's verification step is not smart enough to see
        // this, and may complain otherwise.
        if (DEBUG) System.out.println("Initialization marking: ");

        codeAttribute.instructionsAccept(clazz, method,
            necessaryInstructionFilter(
            variableInitializationMarker));

        if (DEBUG) System.out.println();


        // Mark produced stack entries, in order to keep the stack consistent.
        if (DEBUG) System.out.println("Stack consistency fixing:");

        maxMarkedOffset = codeLength - 1;

        while (maxMarkedOffset >= 0)
        {
            int offset = maxMarkedOffset;

            maxMarkedOffset = offset - 1;

            if (partialEvaluator.isTraced(offset))
            {
                Instruction instruction = InstructionFactory.create(codeAttribute.code,
                                                                    offset);

                instruction.accept(clazz, method, codeAttribute, offset, stackConsistencyMarker);

                // Check if this instruction is a branch origin from a branch
                // that straddles some marked code.
                markStraddlingBranches(offset,
                                       partialEvaluator.branchTargets(offset),
                                       true);

                // Check if this instruction is a branch target from a branch
                // that straddles some marked code.
                markStraddlingBranches(offset,
                                       partialEvaluator.branchOrigins(offset),
                                       false);
            }
        }
        if (DEBUG) System.out.println();


        // Mark unnecessary popping instructions, in order to keep the stack
        // consistent.
        if (DEBUG) System.out.println("Extra pop marking:");

        maxMarkedOffset = codeLength - 1;

        while (maxMarkedOffset >= 0)
        {
            int offset = maxMarkedOffset;

            maxMarkedOffset = offset - 1;

            if (partialEvaluator.isTraced(offset) &&
                !isInstructionNecessary(offset))
            {
                Instruction instruction = InstructionFactory.create(codeAttribute.code,
                                                                    offset);

                instruction.accept(clazz, method, codeAttribute, offset, extraPushPopInstructionMarker);

                // Check if this instruction is a branch origin from a branch
                // that straddles some marked code.
                markStraddlingBranches(offset,
                                       partialEvaluator.branchTargets(offset),
                                       true);

                // Check if this instruction is a branch target from a branch
                // that straddles some marked code.
                markStraddlingBranches(offset,
                                       partialEvaluator.branchOrigins(offset),
                                       false);
            }
        }
        if (DEBUG) System.out.println();


        if (DEBUG_RESULTS)
        {
            System.out.println("Instruction usage results:");

            int offset = 0;
            do
            {
                Instruction instruction = InstructionFactory.create(codeAttribute.code,
                                                                    offset);
                System.out.println((isInstructionNecessary(offset)             ? " + " :
                                    isExtraPushPopInstructionNecessary(offset) ? " ~ " :
                                                                                 " - ") +
                                   instruction.toString(offset));

                offset += instruction.length(offset);
            }
            while (offset < codeLength);
        }
    }


    /**
     * This MemberVisitor marks stack entries that aren't necessary because
     * parameters aren't used in the methods that are visited.
     */
    private class MyParameterUsageMarker
    extends       SimplifiedVisitor
    implements    InstructionVisitor,
                  ConstantVisitor,
                  MemberVisitor
    {
        private int  parameterSize;
        private long usedParameters;


        // Implementations for InstructionVisitor.

        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


        public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
        {
            switch (constantInstruction.opcode)
            {
                case InstructionConstants.OP_INVOKEVIRTUAL:
                case InstructionConstants.OP_INVOKESPECIAL:
                case InstructionConstants.OP_INVOKESTATIC:
                case InstructionConstants.OP_INVOKEINTERFACE:
                {
                    parameterSize = 0;
                    clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);

                    // Mark unused parameters.
                    for (int index = 0; index < parameterSize; index++)
                    {
                        if (index < 64 &&
                            (usedParameters & (1L << index)) == 0L)
                        {
                            TracedStack stack =
                                partialEvaluator.getStackBefore(offset);

                            int stackIndex = stack.size() - parameterSize + index;

                            if (DEBUG)
                            {
                                System.out.println("  ["+offset+"] Ignoring parameter #"+index+" (stack entry #"+stackIndex+" ["+stack.getBottom(stackIndex)+"])");
                                System.out.println("    Full stack: "+stack);
                            }

                            markStackEntryUnwantedBefore(offset, stackIndex);
                        }
                    }
                    break;
                }
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
            // Get the total size of the parameters and the mask of the used
            // parameters.
            parameterSize  = ParameterUsageMarker.getParameterSize(programMethod);
            usedParameters = ParameterUsageMarker.getUsedParameters(programMethod);
        }
    }


    /**
     * This InstructionVisitor marks the instructions that are intrinsically
     * necessary, because they have side effects.
     */
    private class MyInitialUsageMarker
    extends       SimplifiedVisitor
    implements    InstructionVisitor,
                  ConstantVisitor,
                  ParameterVisitor
    {
        private final MemberVisitor reverseDependencyCreator = new AllParameterVisitor(true, this);

        // Parameters and values for visitor methods.
        private int    referencingOffset;
        private int    referencingPopCount;


        // Implementations for InstructionVisitor.

        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
        {
            if (sideEffectInstructionChecker.hasSideEffects(clazz,
                                                            method,
                                                            codeAttribute,
                                                            offset,
                                                            instruction))
            {
                markInstruction(offset);
            }
        }


        public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
        {
            switch (simpleInstruction.opcode)
            {
                case InstructionConstants.OP_IASTORE:
                case InstructionConstants.OP_LASTORE:
                case InstructionConstants.OP_FASTORE:
                case InstructionConstants.OP_DASTORE:
                case InstructionConstants.OP_AASTORE:
                case InstructionConstants.OP_BASTORE:
                case InstructionConstants.OP_CASTORE:
                case InstructionConstants.OP_SASTORE:
                    createReverseDependencies(clazz, offset, simpleInstruction);

                    // Also check for side-effects of the instruction itself.
                    visitAnyInstruction(clazz, method, codeAttribute, offset, simpleInstruction);
                    break;

                default:
                    visitAnyInstruction(clazz, method, codeAttribute, offset, simpleInstruction);
                    break;
            }
        }


        public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
        {
            switch (constantInstruction.opcode)
            {
                case InstructionConstants.OP_ANEWARRAY:
                case InstructionConstants.OP_MULTIANEWARRAY:
                    // We may have to mark the instruction due to initializers.
                    referencingOffset = offset;
                    clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);

                    // Also check for side-effects of the instruction itself.
                    visitAnyInstruction(clazz, method, codeAttribute, offset, constantInstruction);
                    break;

                case InstructionConstants.OP_LDC:
                case InstructionConstants.OP_LDC_W:
                case InstructionConstants.OP_NEW:
                case InstructionConstants.OP_GETSTATIC:
                    // We may have to mark the instruction due to initializers.
                    referencingOffset = offset;
                    clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                    break;

                case InstructionConstants.OP_PUTFIELD:
                    // We generally have to mark the putfield instruction,
                    // unless it's never read. We can reverse the dependencies
                    // if it's a field of a recently created instance.
                    if (sideEffectInstructionChecker.hasSideEffects(clazz,
                                                                    method,
                                                                    codeAttribute,
                                                                    offset,
                                                                    constantInstruction))
                    {
                        createReverseDependencies(clazz, offset, constantInstruction);
                    }
                    break;

                case InstructionConstants.OP_INVOKEVIRTUAL:
                case InstructionConstants.OP_INVOKESPECIAL:
                case InstructionConstants.OP_INVOKESTATIC:
                case InstructionConstants.OP_INVOKEINTERFACE:
                    referencingOffset   = offset;
                    referencingPopCount = constantInstruction.stackPopCount(clazz);
                    clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                    break;

                default:
                    visitAnyInstruction(clazz, method, codeAttribute, offset, constantInstruction);
                    break;
            }
        }


        public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
        {
            if (branchInstruction.opcode == InstructionConstants.OP_GOTO &&
                branchInstruction.branchOffset == 0)
            {
                if (DEBUG) System.out.print("(infinite loop)");
                markInstruction(offset);
            }
            else
            {
                visitAnyInstruction(clazz, method, codeAttribute, offset, branchInstruction);
            }
        }


        // Implementations for ConstantVisitor.

        public void visitAnyConstant(Clazz clazz, Constant constant) {}


        public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
        {
            Clazz referencedClass = stringConstant.referencedClass;

            // If a static initializer may have side effects, the instruction
            // has to be marked.
            if (referencedClass != null &&
                SideEffectClassChecker.mayHaveSideEffects(clazz,
                                                          referencedClass))
            {
                // Mark the invocation.
                markInstruction(referencingOffset);
            }
        }


        public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
        {
            Clazz referencedClass = classConstant.referencedClass;

            // If a static initializer may have side effects, the instruction
            // has to be marked.
            if (referencedClass == null ||
                SideEffectClassChecker.mayHaveSideEffects(clazz,
                                                          referencedClass))
            {
                // Mark the invocation.
                markInstruction(referencingOffset);
            }
        }


        public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
        {
            clazz.constantPoolEntryAccept(fieldrefConstant.u2classIndex, this);
        }


        public void visitAnyMethodrefConstant(Clazz clazz, RefConstant refConstant)
        {
            Method referencedMethod = (Method)refConstant.referencedMember;

//            if (referencedMethod != null)
//            {
//                System.out.println("InstructionUsageMarker$MyInitialUsageMarker.visitAnyMethodrefConstant [" + refConstant.getClassName(clazz) + "." + refConstant.getName(clazz) +
//                                   "]: mark! esc = " + ParameterEscapeMarker.getEscapingParameters(referencedMethod) +
//                                   ", mod = " + ParameterEscapeMarker.modifiesAnything(referencedMethod) +
//                                   ", side = " + SideEffectClassChecker.mayHaveSideEffects(clazz,
//                                                                                           refConstant.referencedClass,
//                                                                                           referencedMethod));
//            }

            // Is the method invocation really necessary?
            if (SideEffectInstructionChecker.OPTIMIZE_CONSERVATIVELY     &&
                referencedMethod != null                                 &&
                SideEffectMethodMarker.hasSideEffects(referencedMethod)  &&
                // Skip if the method was explicitly marked as having no external side-effects.
                !NoExternalSideEffectMethodMarker.hasNoExternalSideEffects(referencedMethod))
            {
                // In case we shall optimize conservatively, always mark the method
                // call if the referenced method has side effects.
                markInstruction(referencingOffset);
            }
            else if (referencedMethod == null                                       ||
                ParameterEscapeMarker.getEscapingParameters(referencedMethod) != 0L ||
                ParameterEscapeMarker.modifiesAnything(referencedMethod)            ||
                SideEffectClassChecker.mayHaveSideEffects(clazz,
                                                          refConstant.referencedClass,
                                                          referencedMethod))
            {
//                System.out.println("  -> mark ["+referencingOffset+"]");
                // Mark the invocation.
                markInstruction(referencingOffset);
            }
            else
            {
                if (DEBUG)
                {
                    System.out.println("  ["+referencingOffset+"] Checking parameters of ["+refConstant.getClassName(clazz)+"."+refConstant.getName(clazz)+refConstant.getType(clazz)+"] (pop count = "+referencingPopCount+")");
                }

                // Create reverse dependencies for reference parameters that
                // are modified.
                refConstant.referencedMemberAccept(reverseDependencyCreator);
            }
        }


        // Implementations for ParameterVisitor.

        public void visitParameter(Clazz clazz, Member member, int parameterIndex, int parameterCount, int parameterOffset, int parameterSize, String parameterType, Clazz referencedClass)
        {
            Method method = (Method)member;

            if (DEBUG)
            {
                System.out.println("    P"+parameterIndex+
                                   ": escaping = "+ParameterEscapeMarker.isParameterEscaping(method, parameterIndex)+
                                   ", modified = "+ParameterEscapeMarker.isParameterModified(method, parameterIndex)+
                                   ", returned = "+ParameterEscapeMarker.isParameterReturned(method, parameterIndex));
            }

            // Create a reverse dependency if the reference parameter is
            // modified.
            if (ParameterEscapeMarker.isParameterModified(method, parameterIndex))
            {
                createReverseDependencies(referencingOffset,
                                          parameterSize - parameterOffset - 1);
            }
        }


        /**
         * Marks the specified instruction offset or creates reverse
         * dependencies to the producers of its bottom popped stack entry.
         */
        private void createReverseDependencies(Clazz       clazz,
                                               int         offset,
                                               Instruction instruction)
        {
            createReverseDependencies(offset,
                                      instruction.stackPopCount(clazz) - 1);
        }


        /**
         * Marks the specified instruction offset or creates reverse
         * dependencies to the producers of the specified stack entry, if it
         * is a reference value.
         */
        private void createReverseDependencies(int offset,
                                               int stackEntryIndex)
        {
            TracedStack stackBefore = partialEvaluator.getStackBefore(offset);
            Value       stackEntry  = stackBefore.getTop(stackEntryIndex);
//            System.out.println("     ["+offset+"] s"+stackEntryIndex+": ["+stackEntry+"]");

            if (stackEntry.computationalType() == Value.TYPE_REFERENCE)
            {
                ReferenceValue referenceValue = stackEntry.referenceValue();
//                System.out.println("EvaluationShrinker$MyInitialUsageMarker.createReverseDependencies: ["+offset+"] ["+referenceValue+"]?");
                // The null reference value may not have a trace value.
                if (referenceValue.isNull() != Value.ALWAYS)
                {
                    if (referenceValue instanceof TracedReferenceValue)
                    {
                        TracedReferenceValue tracedReferenceValue =
                            (TracedReferenceValue)referenceValue;

                        createReverseDependencies(offset,
                                                  tracedReferenceValue.getTraceValue().instructionOffsetValue());
                    }
                    else
                    {
//                        System.out.println("InstructionUsageMarker$MyInitialUsageMarker.createReverseDependencies: not a TracedReferenceValue");
                        markInstruction(offset);
                    }
                }
            }
        }


        /**
         * Marks the specified instruction offset or creates reverse
         * dependencies to the producers of the given reference value.
         */
        private void createReverseDependencies(int                    offset,
                                               InstructionOffsetValue producerOffsets)
        {
            InstructionOffsetValue consumerOffset =
                new InstructionOffsetValue(offset);

            int offsetCount = producerOffsets.instructionOffsetCount();
            for (int offsetIndex = 0; offsetIndex < offsetCount; offsetIndex++)
            {
                if (producerOffsets.isNewinstance(offsetIndex))
                {
                    // Create a reverse dependency. If the creating instruction
                    // is necessary, then so is this one.
                    int producerOffset = producerOffsets.instructionOffset(offsetIndex);

                    // Avoid circular dependencies in code that loops with
                    // instances on the stack (like the string encryption code).
                    if (producerOffset != offset)
                    {
                        if (DEBUG) System.out.println("  Inserting reverse dependency from instance producers ["+producerOffset+"] to ["+offset+"]");

                        InstructionOffsetValue reverseDependency =
                            reverseDependencies[producerOffset];

                        reverseDependencies[producerOffset] =
                            reverseDependency == null ?
                                consumerOffset :
                                reverseDependency.generalize(consumerOffset);
                    }
                }
                else
                {
                    // Just mark the instruction.
                    markInstruction(offset);
                }
            }
        }
    }


    /**
     * This InstructionVisitor marks the producing instructions and produced
     * variables and stack entries of the instructions that it visits.
     * Simplified method arguments are ignored.
     */
    private class MyProducerMarker
    extends       SimplifiedVisitor
    implements    InstructionVisitor
    {
        // Implementations for InstructionVisitor.

        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
        {
            markStackProducers(clazz, offset, instruction);
        }


        public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
        {
            switch (simpleInstruction.opcode)
            {
                case InstructionConstants.OP_DUP:
                    conditionallyMarkStackEntryProducers(offset, 0, 0);
                    conditionallyMarkStackEntryProducers(offset, 1, 0);
                    break;
                case InstructionConstants.OP_DUP_X1:
                    conditionallyMarkStackEntryProducers(offset, 0, 0);
                    conditionallyMarkStackEntryProducers(offset, 1, 1);
                    conditionallyMarkStackEntryProducers(offset, 2, 0);
                    break;
                case InstructionConstants.OP_DUP_X2:
                    conditionallyMarkStackEntryProducers(offset, 0, 0);
                    conditionallyMarkStackEntryProducers(offset, 1, 1);
                    conditionallyMarkStackEntryProducers(offset, 2, 2);
                    conditionallyMarkStackEntryProducers(offset, 3, 0);
                    break;
                case InstructionConstants.OP_DUP2:
                    conditionallyMarkStackEntryProducers(offset, 0, 0);
                    conditionallyMarkStackEntryProducers(offset, 1, 1);
                    conditionallyMarkStackEntryProducers(offset, 2, 0);
                    conditionallyMarkStackEntryProducers(offset, 3, 1);
                    break;
                case InstructionConstants.OP_DUP2_X1:
                    conditionallyMarkStackEntryProducers(offset, 0, 0);
                    conditionallyMarkStackEntryProducers(offset, 1, 1);
                    conditionallyMarkStackEntryProducers(offset, 2, 2);
                    conditionallyMarkStackEntryProducers(offset, 3, 0);
                    conditionallyMarkStackEntryProducers(offset, 4, 1);
                    break;
                case InstructionConstants.OP_DUP2_X2:
                    conditionallyMarkStackEntryProducers(offset, 0, 0);
                    conditionallyMarkStackEntryProducers(offset, 1, 1);
                    conditionallyMarkStackEntryProducers(offset, 2, 2);
                    conditionallyMarkStackEntryProducers(offset, 3, 3);
                    conditionallyMarkStackEntryProducers(offset, 4, 0);
                    conditionallyMarkStackEntryProducers(offset, 5, 1);
                    break;
                case InstructionConstants.OP_SWAP:
                    conditionallyMarkStackEntryProducers(offset, 0, 1);
                    conditionallyMarkStackEntryProducers(offset, 1, 0);
                    break;
                default:
                    markStackProducers(clazz, offset, simpleInstruction);
                    break;
            }
        }


        public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
        {
            // Is the variable being loaded or incremented?
            if (variableInstruction.isLoad())
            {
                markVariableProducers(offset, variableInstruction.variableIndex);
            }
            else
            {
                markStackProducers(clazz, offset, variableInstruction);
            }
        }


        public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
        {
            markStackProducers(clazz, offset, constantInstruction);
        }


        public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
        {
            // Explicitly mark the produced stack entry of a 'jsr' instruction,
            // because the consuming 'astore' instruction of the subroutine is
            // cleared every time it is traced.
            if (branchInstruction.opcode == InstructionConstants.OP_JSR ||
                branchInstruction.opcode == InstructionConstants.OP_JSR_W)
            {
                markStackEntryAfter(offset, 0);
            }
            else
            {
                markStackProducers(clazz, offset, branchInstruction);
            }
        }
    }


    /**
     * This InstructionVisitor marks variable initializations that are
     * necessary to appease the JVM.
     */
    private class MyVariableInitializationMarker
    extends       SimplifiedVisitor
    implements    InstructionVisitor
    {
        // Implementations for InstructionVisitor.

        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


        public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
        {
            // Is the variable being loaded or incremented?
            if (variableInstruction.isLoad())
            {
                // Mark any variable initializations for this variable load that
                // are required according to the JVM.
                markVariableInitializersBefore(offset, variableInstruction.variableIndex, null);
            }
        }
    }


    /**
     * This InstructionVisitor marks stack entries that should be pushed
     * (and previously unnecessary pushing instructions) to keep the stack
     * consistent at later points in the execution.
     */
    private class MyStackConsistencyMarker
    extends       SimplifiedVisitor
    implements    InstructionVisitor
    {
        // Implementations for InstructionVisitor.

        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
        {
            // We check all entries to make sure the stack is also consistent
            // at method exit points, where some stack entries might be
            // discarded.
            int stackSize = partialEvaluator.getStackBefore(offset).size();

            for (int stackIndex = 0; stackIndex < stackSize; stackIndex++)
            {
                // Is this stack entry pushed by any producer
                // (because it is required by other consumers)?
                if (!isStackEntryUnwantedBefore(offset, stackIndex) &&
                    isStackEntryPresentBefore(offset, stackIndex))
                {
                    // Mark all produced stack entries.
                    markStackEntryProducers(offset, stackIndex, false);
                }
            }
        }
    }


    /**
     * This InstructionVisitor marks instructions that should still push or
     * pop some values to keep the stack consistent.
     */
    private class MyExtraPushPopInstructionMarker
    extends       SimplifiedVisitor
    implements    InstructionVisitor
    {
        // Implementations for InstructionVisitor.

        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
        {
            // Check all stack entries that are popped.
            //
            // Typical case: a stack value that is required elsewhere or a
            // pushed exception type that still has to be popped.
            int stackSize = partialEvaluator.getStackBefore(offset).size();

            int firstStackIndex =
                    stackSize - instruction.stackPopCount(clazz);

            for (int stackIndex = firstStackIndex; stackIndex < stackSize; stackIndex++)
            {
                // Is this stack entry pushed by any producer
                // (because it is required by other consumers)?
                if (!isStackEntryUnwantedBefore(offset, stackIndex) &&
                    isStackEntryPresentBefore(offset, stackIndex))
                {
                    // Mark that we'll need an extra pop instruction.
                    markExtraPushPopInstruction(offset);

                    // [DGD-481][DGD-504] Mark the stack entries and
                    // their producers again for a push/pop. In Kotlin
                    // code, it can happen that we have missed a producer
                    // during stack consistency marking.
                    markStackEntryProducers(offset, stackIndex, false);
                }
            }
        }
    }


    // Small utility methods.

    /**
     * Marks the producing instructions of the variable consumer at the given
     * offset.
     * @param consumerOffset the offset of the variable consumer.
     * @param variableIndex  the index of the variable that is loaded.
     */
    private void markVariableProducers(int consumerOffset,
                                       int variableIndex)
    {
        InstructionOffsetValue producerOffsets =
            partialEvaluator.getVariablesBefore(consumerOffset).getProducerValue(variableIndex).instructionOffsetValue();

        if (producerOffsets != null)
        {
            int offsetCount = producerOffsets.instructionOffsetCount();
            for (int offsetIndex = 0; offsetIndex < offsetCount; offsetIndex++)
            {
                if (!producerOffsets.isMethodParameter(offsetIndex) &&
                    !producerOffsets.isExceptionHandler(offsetIndex))
                {
                    // Make sure the variable and the instruction are marked
                    // at the producing offset.
                    int offset = producerOffsets.instructionOffset(offsetIndex);

                    markInstruction(offset);
                }
            }
        }
    }


    /**
     * Ensures that the given variable is initialized before the specified
     * consumer of that variable, in the JVM's view.
     * @param consumerOffset the instruction offset before which the variable
     *                       needs to be initialized.
     * @param variableIndex  the index of the variable.
     * @param visitedOffsets the already visited consumer offsets, needed to
     *                       prevent infinite loops.
     * @return the updated visited consumer offsets.
     */
    private InstructionOffsetValue markVariableInitializersBefore(int                    consumerOffset,
                                                                  int                    variableIndex,
                                                                  InstructionOffsetValue visitedOffsets)
    {
        // Avoid infinite loops by stopping recursion if we encounter
        // an already visited offset.
        if (visitedOffsets == null ||
            !visitedOffsets.contains(consumerOffset))
        {
            visitedOffsets = visitedOffsets == null ?
                new InstructionOffsetValue(consumerOffset) :
                visitedOffsets.add(consumerOffset);

            // Make sure the variable is initialized after all producers.
            // Use the simple evaluator, to get the JVM's view of what is
            // initialized.
            InstructionOffsetValue producerOffsets =
                simplePartialEvaluator.getVariablesBefore(consumerOffset).getProducerValue(variableIndex).instructionOffsetValue();

            int offsetCount = producerOffsets.instructionOffsetCount();
            for (int offsetIndex = 0; offsetIndex < offsetCount; offsetIndex++)
            {
                if (!producerOffsets.isMethodParameter(offsetIndex) &&
                    !producerOffsets.isExceptionHandler(offsetIndex))
                {
                    int producerOffset =
                        producerOffsets.instructionOffset(offsetIndex);

                    visitedOffsets =
                        markVariableInitializersAfter(producerOffset,
                                                      variableIndex,
                                                      visitedOffsets);
                }
            }
        }

        return visitedOffsets;
    }


    /**
     * Ensures that the given variable is initialized after the specified
     * producer of that variable, in the JVM's view.
     * @param producerOffset the instruction offset after which the variable
     *                       needs to be initialized.
     * @param variableIndex  the index of the variable.
     * @param visitedOffsets the already visited consumer offsets, needed to
     *                       prevent infinite loops.
     * @return the updated visited consumer offsets.
     */
    private InstructionOffsetValue markVariableInitializersAfter(int                    producerOffset,
                                                                 int                    variableIndex,
                                                                 InstructionOffsetValue visitedOffsets)
    {
        // No problem if the producer has already been marked.
        if (!isInstructionNecessary(producerOffset))
        {
            // Is the unmarked producer a variable initialization?
            if (isVariableInitialization(producerOffset, variableIndex))
            {
                // Mark the producer.
                if (DEBUG) System.out.print("  Marking initialization of v"+variableIndex+" at ");

                markInstruction(producerOffset);

                if (DEBUG) System.out.println();
            }
            else
            {
                // Don't mark the producer, but recursively look at the
                // preceding producers of the same variable. Their values
                // will fall through, replacing this producer.
                visitedOffsets =
                    markVariableInitializersBefore(producerOffset,
                                                   variableIndex,
                                                   visitedOffsets);
            }
        }

        return visitedOffsets;
    }


    /**
     * Marks the stack entries and their producing instructions of the
     * consumer at the given offset.
     * @param clazz          the containing class.
     * @param consumerOffset the offset of the consumer.
     * @param consumer       the consumer of the stack entries.
     */
    private void markStackProducers(Clazz       clazz,
                                    int         consumerOffset,
                                    Instruction consumer)
    {
        TracedStack tracedStack =
            partialEvaluator.getStackBefore(consumerOffset);

        int stackSize = tracedStack.size();

        // Mark the producers of the popped values.
        int popCount = consumer.stackPopCount(clazz);
        for (int stackIndex = stackSize - popCount; stackIndex < stackSize; stackIndex++)
        {
            markStackEntryProducers(consumerOffset, stackIndex, true);
        }
    }


    /**
     * Marks the stack entry and the corresponding producing instructions
     * of the consumer at the given offset, if the stack entry of the
     * consumer is marked.
     * @param consumerOffset        the offset of the consumer.
     * @param consumerTopStackIndex the index of the stack entry to be checked
     *                              (counting from the top).
     * @param producerTopStackIndex the index of the stack entry to be marked
     *                              (counting from the top).
     */
    private void conditionallyMarkStackEntryProducers(int consumerOffset,
                                                      int consumerTopStackIndex,
                                                      int producerTopStackIndex)
    {
        int consumerBottomStackIndex = partialEvaluator.getStackAfter(consumerOffset).size() - consumerTopStackIndex - 1;

        if (isStackEntryNecessaryAfter(consumerOffset, consumerBottomStackIndex))
        {
            int producerBottomStackIndex = partialEvaluator.getStackBefore(consumerOffset).size() - producerTopStackIndex - 1;

            markStackEntryProducers(consumerOffset, producerBottomStackIndex, true);
        }
    }


    /**
     * Marks the stack entry and optionally the corresponding producing
     * instructions of the consumer at the given offset.
     * @param consumerOffset   the offset of the consumer.
     * @param stackIndex       the index of the stack entry to be marked
     *                         (counting from the bottom).
     * @param markInstructions specifies whether the producing instructions
     *                         should be marked.
     */
    private void markStackEntryProducers(int     consumerOffset,
                                         int     stackIndex,
                                         boolean markInstructions)
    {
        if (!isStackEntryUnwantedBefore(consumerOffset, stackIndex))
        {
            markStackEntryProducers(partialEvaluator.getStackBefore(consumerOffset).getBottomProducerValue(stackIndex).instructionOffsetValue(),
                                    stackIndex,
                                    markInstructions);
        }
    }


    /**
     * Marks the stack entry and optionally its producing instructions at the
     * given offsets.
     * @param producerOffsets  the offsets of the producers to be marked.
     * @param stackIndex       the index of the stack entry to be marked
     *                         (counting from the bottom).
     * @param markInstructions specifies whether the producing instructions
     *                         should be marked.
     */
    private void markStackEntryProducers(InstructionOffsetValue producerOffsets,
                                         int                    stackIndex,
                                         boolean                markInstructions)
    {
        if (producerOffsets != null)
        {
            int offsetCount = producerOffsets.instructionOffsetCount();
            for (int offsetIndex = 0; offsetIndex < offsetCount; offsetIndex++)
            {
                if (!producerOffsets.isExceptionHandler(offsetIndex))
                {
                    // Make sure the stack entry and the instruction are marked
                    // at the producing offset.
                    int offset = producerOffsets.instructionOffset(offsetIndex);

                    markStackEntryAfter(offset, stackIndex);

                    if (markInstructions)
                    {
                        // We can mark the producer.
                        markInstruction(offset);
                    }
                    else
                    {
                        // We'll need to push a stack entry at that point
                        // instead.
                        markExtraPushPopInstruction(offset);
                    }
                }
            }
        }
    }


    /**
     * Marks any modification instructions that are required by the specified
     * creation instruction (new, newarray, method returning new
     * instance,...), so this new instance is properly initialized.
     * @param instructionOffset the offset of the creation instruction.
     */
    private void markReverseDependencies(int instructionOffset)
    {
        InstructionOffsetValue reverseDependency =
            reverseDependencies[instructionOffset];

        if (reverseDependency != null)
        {
            markInstructions(reverseDependency);
        }
    }


    /**
     * Marks the branch instructions of straddling branches, if they straddle
     * some code that has been marked.
     * @param instructionOffset   the offset of the branch origin or branch target.
     * @param branchOffsets       the offsets of the straddling branch targets
     *                            or branch origins.
     * @param isPointingToTargets <code>true</code> if the above offsets are
     *                            branch targets, <code>false</code> if they
     *                            are branch origins.
     */
    private void markStraddlingBranches(int                    instructionOffset,
                                        InstructionOffsetValue branchOffsets,
                                        boolean                isPointingToTargets)
    {
        if (branchOffsets != null)
        {
            // Loop over all branch offsets.
            int branchCount = branchOffsets.instructionOffsetCount();
            for (int branchIndex = 0; branchIndex < branchCount; branchIndex++)
            {
                // Is the branch straddling forward any necessary instructions?
                int branchOffset = branchOffsets.instructionOffset(branchIndex);

                // Is the offset pointing to a branch origin or to a branch target?
                if (isPointingToTargets)
                {
                    markStraddlingBranch(instructionOffset,
                                         branchOffset,
                                         instructionOffset,
                                         branchOffset);
                }
                else
                {
                    markStraddlingBranch(instructionOffset,
                                         branchOffset,
                                         branchOffset,
                                         instructionOffset);
                }
            }
        }
    }


    private void markStraddlingBranch(int instructionOffsetStart,
                                      int instructionOffsetEnd,
                                      int branchOrigin,
                                      int branchTarget)
    {
        if (!isInstructionNecessary(branchOrigin) &&
            isAnyInstructionNecessary(instructionOffsetStart, instructionOffsetEnd))
        {
            if (DEBUG) System.out.print("["+branchOrigin+"->"+branchTarget+"]");

            // Mark the branch instruction.
            markInstruction(branchOrigin);
        }
    }


    /**
     * Initializes the necessary data structure.
     */
    private void initializeNecessary(CodeAttribute codeAttribute)
    {
        int codeLength = codeAttribute.u4codeLength;
        int maxLocals  = codeAttribute.u2maxLocals;
        int maxStack   = codeAttribute.u2maxStack;

        // Create new arrays for storing information at each instruction offset.
        reverseDependencies =
            ArrayUtil.ensureArraySize(reverseDependencies, codeLength, null);

        if (stacksNecessaryAfter.length    < codeLength ||
            stacksNecessaryAfter[0].length < maxStack)
        {
            stacksNecessaryAfter = new boolean[codeLength][maxStack];
        }
        else
        {
            for (int offset = 0; offset < codeLength; offset++)
            {
                Arrays.fill(stacksNecessaryAfter[offset], 0, maxStack, false);
            }
        }

        if (stacksUnwantedBefore.length    < codeLength ||
            stacksUnwantedBefore[0].length < maxStack)
        {
            stacksUnwantedBefore = new boolean[codeLength][maxStack];
        }
        else
        {
            for (int offset = 0; offset < codeLength; offset++)
            {
                Arrays.fill(stacksUnwantedBefore[offset], 0, maxStack, false);
            }
        }

        instructionsNecessary =
            ArrayUtil.ensureArraySize(instructionsNecessary,
                                      codeLength,
                                      false);

        extraPushPopInstructionsNecessary =
            ArrayUtil.ensureArraySize(extraPushPopInstructionsNecessary,
                                      codeLength,
                                      false);
    }


    /**
     * Returns whether the specified variable is initialized at the specified
     * offset.
     */
    private boolean isVariableInitialization(int instructionOffset,
                                             int variableIndex)
    {
        // Wasn't the variable set yet?
        Value valueBefore = simplePartialEvaluator.getVariablesBefore(instructionOffset).getValue(variableIndex);
        if (valueBefore == null)
        {
            return true;
        }

        // Is the computational type different now?
        Value valueAfter = simplePartialEvaluator.getVariablesAfter(instructionOffset).getValue(variableIndex);
        if (valueAfter.computationalType() != valueBefore.computationalType())
        {
            return true;
        }

        // Is the reference type different now?
        if (valueAfter.computationalType() == Value.TYPE_REFERENCE &&
            (valueAfter.referenceValue().isNull() == Value.ALWAYS ||
             !valueAfter.referenceValue().getType().equals(valueBefore.referenceValue().getType())))
        {
            return true;
        }

        // Was the producer an argument (which may be removed)?
        InstructionOffsetValue producersBefore = simplePartialEvaluator.getVariablesBefore(instructionOffset).getProducerValue(variableIndex).instructionOffsetValue();
        return producersBefore.instructionOffsetCount() == 1 &&
               producersBefore.isMethodParameter(0);
    }


    /**
     * Marks the stack entry after the given offset.
     * @param instructionOffset the offset of the stack entry to be marked.
     * @param stackIndex        the index of the stack entry to be marked
     *                          (counting from the bottom).
     */
    private void markStackEntryAfter(int instructionOffset,
                                     int stackIndex)
    {
        if (!isStackEntryNecessaryAfter(instructionOffset, stackIndex))
        {
            if (DEBUG) System.out.print("["+instructionOffset+".s"+stackIndex+"],");

            stacksNecessaryAfter[instructionOffset][stackIndex] = true;

            if (maxMarkedOffset < instructionOffset)
            {
                maxMarkedOffset = instructionOffset;
            }
        }
    }


    /**
     * Marks the specified stack entry as unwanted, typically because it is
     * an unused parameter of a method invocation.
     * @param instructionOffset the offset of the consumer.
     * @param stackIndex        the index of the stack entry to be marked
     *                          (counting from the bottom).
     */
    private void markStackEntryUnwantedBefore(int instructionOffset,
                                              int stackIndex)
    {
        stacksUnwantedBefore[instructionOffset][stackIndex] = true;
    }


    /**
     * Marks the specified instructions as used.
     * @param instructionOffsets the offsets of the instructions.
     */
    private void markInstructions(InstructionOffsetValue instructionOffsets)
    {
        int count = instructionOffsets.instructionOffsetCount();

        for (int index = 0; index < count; index++)
        {
            markInstruction(instructionOffsets.instructionOffset(index));
        }
    }


    /**
     * Marks the specified instruction as used.
     * @param instructionOffset the offset of the instruction.
     */
    private void markInstruction(int instructionOffset)
    {
        if (!isInstructionNecessary(instructionOffset))
        {
            if (DEBUG) System.out.print(instructionOffset+",");

            instructionsNecessary[instructionOffset] = true;

            if (maxMarkedOffset < instructionOffset)
            {
                maxMarkedOffset = instructionOffset;
            }
        }
    }


    /**
     * Marks that an extra push/pop instruction is required at the given
     * offset, if the current instruction at that offset is unused.
     * @param instructionOffset the offset of the instruction.
     */
    private void markExtraPushPopInstruction(int instructionOffset)
    {
        if (!isInstructionNecessary(instructionOffset) &&
            !isExtraPushPopInstructionNecessary(instructionOffset))
        {
            if (DEBUG) System.out.print(instructionOffset+",");

            extraPushPopInstructionsNecessary[instructionOffset] = true;

            if (maxMarkedOffset < instructionOffset)
            {
                maxMarkedOffset = instructionOffset;
            }
        }
    }


    /**
     * Returns whether any instruction in the specified sequence of
     * instructions is necessary.
     * @param startInstructionOffset the start offset of the instruction
     *                               sequence (inclusive).
     * @param endInstructionOffset   the end offset of the instruction
     *                               sequence (exclusive).
     * @return whether any instruction is necessary.
     */
    private boolean isAnyInstructionNecessary(int startInstructionOffset,
                                              int endInstructionOffset)
    {
        for (int instructionOffset = startInstructionOffset;
             instructionOffset < endInstructionOffset;
             instructionOffset++)
        {
            if (isInstructionNecessary(instructionOffset) ||
                isExtraPushPopInstructionNecessary(instructionOffset))
            {
                return true;
            }
        }

        return false;
    }


   /**
     * This InstructionVisitor delegates its visits to a given
     * InstructionVisitor, but only if the instruction has been marked as
     * necessary (or not).
     */
   private class MyNecessaryInstructionFilter implements InstructionVisitor
   {
       private final boolean            necessary;
       private final InstructionVisitor instructionVisitor;


       public MyNecessaryInstructionFilter(boolean            necessary,
                                           InstructionVisitor instructionVisitor)
       {
           this.necessary          = necessary;
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


       // Small utility methods.

       /**
        * Returns whether the instruction at the given offset should be
        * visited, depending on whether it is necessary or not.
        */
       private boolean shouldVisit(int offset)
       {
           return isInstructionNecessary(offset) == necessary;
       }
   }
}