/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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
package proguard.optimize.peephole;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.evaluation.BranchTargetFinder;

/**
 * This InstructionVisitor replaces a given pattern instruction sequence by
 * another given replacement instruction sequence. The arguments of the
 * instruction sequences can be wildcards that are matched and replaced.
 *
 * @see InstructionSequenceMatcher
 * @author Eric Lafortune
 */
public class InstructionSequenceReplacer
extends      SimplifiedVisitor
implements   InstructionVisitor,
             ConstantVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    public  static       boolean DEBUG = true;
    //*/

    public static final int X = InstructionSequenceMatcher.X;
    public static final int Y = InstructionSequenceMatcher.Y;
    public static final int Z = InstructionSequenceMatcher.Z;

    public static final int A = InstructionSequenceMatcher.A;
    public static final int B = InstructionSequenceMatcher.B;
    public static final int C = InstructionSequenceMatcher.C;
    public static final int D = InstructionSequenceMatcher.D;

    private static final int BOOLEAN_STRING = 0x1;
    private static final int CHAR_STRING    = 0x2;
    private static final int INT_STRING     = 0x3;
    private static final int LONG_STRING    = 0x4;
    private static final int FLOAT_STRING   = 0x5;
    private static final int DOUBLE_STRING  = 0x6;
    private static final int STRING_STRING  = 0x7;

    public static final int STRING_A_LENGTH  = 0x20000000;
    public static final int BOOLEAN_A_STRING = 0x20000001;
    public static final int CHAR_A_STRING    = 0x20000002;
    public static final int INT_A_STRING     = 0x20000003;
    public static final int LONG_A_STRING    = 0x20000004;
    public static final int FLOAT_A_STRING   = 0x20000005;
    public static final int DOUBLE_A_STRING  = 0x20000006;
    public static final int STRING_A_STRING  = 0x20000007;
    public static final int BOOLEAN_B_STRING = 0x20000010;
    public static final int CHAR_B_STRING    = 0x20000020;
    public static final int INT_B_STRING     = 0x20000030;
    public static final int LONG_B_STRING    = 0x20000040;
    public static final int FLOAT_B_STRING   = 0x20000050;
    public static final int DOUBLE_B_STRING  = 0x20000060;
    public static final int STRING_B_STRING  = 0x20000070;


    private final InstructionSequenceMatcher instructionSequenceMatcher;
    private final Constant[]                 patternConstants;
    private final Instruction[]              replacementInstructions;
    private final BranchTargetFinder         branchTargetFinder;
    private final CodeAttributeEditor        codeAttributeEditor;
    private final InstructionVisitor         extraInstructionVisitor;

    private final MyReplacementInstructionFactory replacementInstructionFactory = new MyReplacementInstructionFactory();


    /**
     * Creates a new InstructionSequenceReplacer.
     * @param patternConstants        any constants referenced by the pattern
     *                                instruction.
     * @param patternInstructions     the pattern instruction sequence.
     * @param replacementInstructions the replacement instruction sequence.
     * @param branchTargetFinder      a branch target finder that has been
     *                                initialized to indicate branch targets
     *                                in the visited code.
     * @param codeAttributeEditor     a code editor that can be used for
     *                                accumulating changes to the code.
     */
    public InstructionSequenceReplacer(Constant[]          patternConstants,
                                       Instruction[]       patternInstructions,
                                       Instruction[]       replacementInstructions,
                                       BranchTargetFinder  branchTargetFinder,
                                       CodeAttributeEditor codeAttributeEditor)
    {
        this(patternConstants,
             patternInstructions,
             replacementInstructions,
             branchTargetFinder,
             codeAttributeEditor,
             null);
    }


    /**
     * Creates a new InstructionSequenceReplacer.
     * @param patternConstants        any constants referenced by the pattern
     *                                instruction.
     * @param branchTargetFinder      a branch target finder that has been
     *                                initialized to indicate branch targets
     *                                in the visited code.
     * @param codeAttributeEditor     a code editor that can be used for
     *                                accumulating changes to the code.
     * @param extraInstructionVisitor an optional extra visitor for all deleted
     *                                load instructions.
     */
    public InstructionSequenceReplacer(Constant[]          patternConstants,
                                       Instruction[]       patternInstructions,
                                       Instruction[]       replacementInstructions,
                                       BranchTargetFinder  branchTargetFinder,
                                       CodeAttributeEditor codeAttributeEditor,
                                       InstructionVisitor  extraInstructionVisitor)
    {
        this.instructionSequenceMatcher = new InstructionSequenceMatcher(patternConstants, patternInstructions);
        this.patternConstants           = patternConstants;
        this.replacementInstructions    = replacementInstructions;
        this.branchTargetFinder         = branchTargetFinder;
        this.codeAttributeEditor        = codeAttributeEditor;
        this.extraInstructionVisitor    = extraInstructionVisitor;
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        // Reset the instruction sequence matcher if the instruction is a branch
        // target or if it has already been modified.
        if ((branchTargetFinder != null &&
             branchTargetFinder.isTarget(offset)) ||
            codeAttributeEditor.isModified(offset))
        {
            instructionSequenceMatcher.reset();
        }

        // Try to match the instruction.
        instruction.accept(clazz, method, codeAttribute, offset, instructionSequenceMatcher);

        // Did the instruction sequence match and is it still unmodified?
        if (instructionSequenceMatcher.isMatching() &&
            matchedInstructionsUnmodified())
        {
            if (DEBUG)
            {
                System.out.println("InstructionSequenceReplacer: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]");
                System.out.println("  Matched:");
                for (int index = 0; index < instructionSequenceMatcher.instructionCount(); index++)
                {
                    int matchedOffset = instructionSequenceMatcher.matchedInstructionOffset(index);
                    System.out.println("    "+InstructionFactory.create(codeAttribute.code, matchedOffset).toString(matchedOffset));
                }
                System.out.println("  Replacement:");
                for (int index = 0; index < replacementInstructions.length; index++)
                {
                    int matchedOffset = instructionSequenceMatcher.matchedInstructionOffset(index);
                    System.out.println("    "+replacementInstructionFactory.create(clazz, index).shrink().toString(matchedOffset));
                }
            }

            // Replace the instruction sequence.
            for (int index = 0; index < replacementInstructions.length; index++)
            {
                codeAttributeEditor.replaceInstruction(instructionSequenceMatcher.matchedInstructionOffset(index),
                                                       replacementInstructionFactory.create(clazz, index));
            }

            // Delete any remaining instructions in the from sequence.
            for (int index = replacementInstructions.length; index < instructionSequenceMatcher.instructionCount(); index++)
            {
                codeAttributeEditor.deleteInstruction(instructionSequenceMatcher.matchedInstructionOffset(index));
            }

            // Visit the instruction, if required.
            if (extraInstructionVisitor != null)
            {
                instruction.accept(clazz,
                                   method,
                                   codeAttribute,
                                   offset,
                                   extraInstructionVisitor);
            }
        }
    }


    // Small utility methods.

    /**
     * Returns whether the matched pattern instructions haven't been modified
     * before.
     */
    private boolean matchedInstructionsUnmodified()
    {
        for (int index = 0; index < instructionSequenceMatcher.instructionCount(); index++)
        {
            if (codeAttributeEditor.isModified(instructionSequenceMatcher.matchedInstructionOffset(index)))
            {
                return false;
            }
        }

        return true;
    }


    /**
     * This class creates replacement instructions for matched sequences, with
     * any matched arguments filled out.
     */
    private class MyReplacementInstructionFactory
    implements    InstructionVisitor
    {
        private Instruction replacementInstruction;


        /**
         * Creates the replacement instruction for the given index in the
         * instruction sequence.
         */
        public Instruction create(Clazz clazz, int index)
        {
            // Create the instruction.
            replacementInstructions[index].accept(clazz,
                                                  null,
                                                  null,
                                                  instructionSequenceMatcher.matchedInstructionOffset(index),
                                                  this);

            // Return it.
            return replacementInstruction;
        }


        // Implementations for InstructionVisitor.

        public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
        {
            replacementInstruction =
                new SimpleInstruction(simpleInstruction.opcode,
                                      matchedArgument(clazz, simpleInstruction.constant));
        }


        public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
        {
            replacementInstruction =
                new VariableInstruction(variableInstruction.opcode,
                                        instructionSequenceMatcher.matchedArgument(variableInstruction.variableIndex),
                                        instructionSequenceMatcher.matchedArgument(variableInstruction.constant));
        }


        public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
        {
            replacementInstruction =
                new ConstantInstruction(constantInstruction.opcode,
                                        matchedConstantIndex((ProgramClass)clazz,
                                                             constantInstruction.constantIndex),
                                        instructionSequenceMatcher.matchedArgument(constantInstruction.constant));
        }


        public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
        {
            replacementInstruction =
                new BranchInstruction(branchInstruction.opcode,
                                      instructionSequenceMatcher.matchedBranchOffset(offset,
                                                                                     branchInstruction.branchOffset));
        }


        public void visitTableSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, TableSwitchInstruction tableSwitchInstruction)
        {
            replacementInstruction =
                new TableSwitchInstruction(tableSwitchInstruction.opcode,
                                           instructionSequenceMatcher.matchedBranchOffset(offset, tableSwitchInstruction.defaultOffset),
                                           instructionSequenceMatcher.matchedArgument(tableSwitchInstruction.lowCase),
                                           instructionSequenceMatcher.matchedArgument(tableSwitchInstruction.highCase),
                                           instructionSequenceMatcher.matchedJumpOffsets(offset,
                                                                                         tableSwitchInstruction.jumpOffsets));

        }


        public void visitLookUpSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LookUpSwitchInstruction lookUpSwitchInstruction)
        {
            replacementInstruction =
                new LookUpSwitchInstruction(lookUpSwitchInstruction.opcode,
                                            instructionSequenceMatcher.matchedBranchOffset(offset, lookUpSwitchInstruction.defaultOffset),
                                            instructionSequenceMatcher.matchedArguments(lookUpSwitchInstruction.cases),
                                            instructionSequenceMatcher.matchedJumpOffsets(offset, lookUpSwitchInstruction.jumpOffsets));
        }


        /**
         * Returns the matched argument for the given pattern argument.
         */
        private int matchedArgument(Clazz clazz, int argument)
        {
            // Special case: do we have to compute the string length?
            if (argument == STRING_A_LENGTH)
            {
                // Return the string length.
                return clazz.getStringString(instructionSequenceMatcher.matchedArgument(A)).length();
            }

            // Otherwise, just return the matched argument.
            return instructionSequenceMatcher.matchedArgument(argument);
        }


        /**
         * Returns the matched or newly created constant index for the given
         * pattern constant index.
         */
        private int matchedConstantIndex(ProgramClass programClass, int constantIndex)
        {
            // Special case: do we have to create a concatenated string?
            if (constantIndex >= BOOLEAN_A_STRING &&
                constantIndex <= (STRING_A_STRING  | STRING_B_STRING))
            {
                // Create a new string constant and return its index.
                return new ConstantPoolEditor(programClass).addStringConstant(
                    argumentAsString(programClass,  constantIndex        & 0xf, A) +
                    argumentAsString(programClass, (constantIndex >>> 4) & 0xf, B),
                    null,
                    null);
            }

            int matchedConstantIndex =
                instructionSequenceMatcher.matchedConstantIndex(constantIndex);

            // Do we have a matched constant index?
            if (matchedConstantIndex > 0)
            {
                // Return its index.
                return matchedConstantIndex;
            }

            // Otherwise, we still have to create a new constant.
            // This currently only works for constants without any wildcards.
            ProgramClass dummyClass = new ProgramClass();
            dummyClass.constantPool = patternConstants;

            return new ConstantAdder(programClass).addConstant(dummyClass, constantIndex);
        }


        private String argumentAsString(ProgramClass programClass,
                                        int          valueType,
                                        int          argument)
        {
            switch (valueType)
            {
                case BOOLEAN_STRING:
                    return Boolean.toString((instructionSequenceMatcher.wasConstant(argument) ?
                        ((IntegerConstant)(programClass.getConstant(instructionSequenceMatcher.matchedConstantIndex(argument)))).getValue() :
                        instructionSequenceMatcher.matchedArgument(argument)) != 0);

                case CHAR_STRING:
                    return Character.toString((char)(instructionSequenceMatcher.wasConstant(argument) ?
                        ((IntegerConstant)(programClass.getConstant(instructionSequenceMatcher.matchedConstantIndex(argument)))).getValue() :
                        instructionSequenceMatcher.matchedArgument(argument)));

                case INT_STRING:
                    return Integer.toString(instructionSequenceMatcher.wasConstant(argument) ?
                        ((IntegerConstant)(programClass.getConstant(instructionSequenceMatcher.matchedConstantIndex(argument)))).getValue() :
                        instructionSequenceMatcher.matchedArgument(argument));

                case LONG_STRING:
                    return Long.toString(instructionSequenceMatcher.wasConstant(argument) ?
                        ((LongConstant)(programClass.getConstant(instructionSequenceMatcher.matchedConstantIndex(argument)))).getValue() :
                        instructionSequenceMatcher.matchedArgument(argument));

                case FLOAT_STRING:
                    return Float.toString(instructionSequenceMatcher.wasConstant(argument) ?
                        ((FloatConstant)(programClass.getConstant(instructionSequenceMatcher.matchedConstantIndex(argument)))).getValue() :
                        instructionSequenceMatcher.matchedArgument(argument));

                case DOUBLE_STRING:
                    return Double.toString(instructionSequenceMatcher.wasConstant(argument) ?
                        ((DoubleConstant)(programClass.getConstant(instructionSequenceMatcher.matchedConstantIndex(argument)))).getValue() :
                        instructionSequenceMatcher.matchedArgument(argument));

                case STRING_STRING:
                    return
                        programClass.getStringString(instructionSequenceMatcher.matchedConstantIndex(argument));

                default:
                    return "";
            }
        }
    }
}
