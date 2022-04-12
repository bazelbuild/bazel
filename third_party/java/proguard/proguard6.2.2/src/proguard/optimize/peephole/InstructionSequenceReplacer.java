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
package proguard.optimize.peephole;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;

/**
 * This InstructionVisitor replaces a given pattern instruction sequence by
 * another given replacement instruction sequence. The arguments of the
 * instruction sequences can be wildcards that are matched and replaced.
 *
 * The class also supports labels ({@link #label()}) and exception handlers
 * ({@link #catch_(int,int,int)}) in replacement sequences. They provide
 * local branch offsets inside the replacement sequences
 * ({@link Label#offset()}). For example, creating a replacement sequence
 * with the help of {@link InstructionSequenceBuilder}:
 * <code>
 *     final InstructionSequenceReplacer.Label TRY_START = InstructionSequenceReplacer.label();
 *     final InstructionSequenceReplacer.Label TRY_END   = InstructionSequenceReplacer.label();
 *     final InstructionSequenceReplacer.Label CATCH_END = InstructionSequenceReplacer.label();
 *
 *     final InstructionSequenceReplacer.Label CATCH_EXCEPTION =
 *         InstructionSequenceReplacer.catch_(TRY_START.offset(),
 *                                            TRY_END.offset(),
 *                                            constantPoolEditor.addClassConstant("java/lang/Exception", null));
 *
 *     Instructions[] replacementInstructions = builder
 *         .label(TRY_START)
 *         ......
 *         .label(TRY_END)
 *         .goto_(CATCH_END.offset())
 *         .catch_(CATCH_EXCEPTION)
 *         ......
 *         .athrow()
 *         .label(CATCH_END)
 *         ......
 *         .instructions();
 * </code>
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
    public  static       boolean DEBUG = System.getProperty("isr") != null;
    //*/

    public static final int X = InstructionSequenceMatcher.X;
    public static final int Y = InstructionSequenceMatcher.Y;
    public static final int Z = InstructionSequenceMatcher.Z;

    public static final int A = InstructionSequenceMatcher.A;
    public static final int B = InstructionSequenceMatcher.B;
    public static final int C = InstructionSequenceMatcher.C;
    public static final int D = InstructionSequenceMatcher.D;
    public static final int E = InstructionSequenceMatcher.E;
    public static final int F = InstructionSequenceMatcher.F;
    public static final int G = InstructionSequenceMatcher.G;
    public static final int H = InstructionSequenceMatcher.H;
    public static final int I = InstructionSequenceMatcher.I;
    public static final int J = InstructionSequenceMatcher.J;
    public static final int K = InstructionSequenceMatcher.K;
    public static final int L = InstructionSequenceMatcher.L;
    public static final int M = InstructionSequenceMatcher.M;
    public static final int N = InstructionSequenceMatcher.N;
    public static final int O = InstructionSequenceMatcher.O;
    public static final int P = InstructionSequenceMatcher.P;
    public static final int Q = InstructionSequenceMatcher.Q;
    public static final int R = InstructionSequenceMatcher.R;

    private static final int LABEL_FLAG = 0x20000000;

    private static final int BOOLEAN_STRING = 0x1;
    private static final int CHAR_STRING    = 0x2;
    private static final int INT_STRING     = 0x3;
    private static final int LONG_STRING    = 0x4;
    private static final int FLOAT_STRING   = 0x5;
    private static final int DOUBLE_STRING  = 0x6;
    private static final int STRING_STRING  = 0x7;

    // Replacement constants that are derived from matched variables.
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

    private static int labelCounter;


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
     *                                instructions.
     * @param patternInstructions     the pattern instruction sequence.
     * @param replacementConstants    any constants referenced by the
     *                                replacement instructions.
     * @param replacementInstructions the replacement instruction sequence.
     * @param branchTargetFinder      a branch target finder that has been
     *                                initialized to indicate branch targets
     *                                in the visited code.
     * @param codeAttributeEditor     a code editor that can be used for
     *                                accumulating changes to the code.
     */
    public InstructionSequenceReplacer(Constant[]          patternConstants,
                                       Instruction[]       patternInstructions,
                                       Constant[]          replacementConstants,
                                       Instruction[]       replacementInstructions,
                                       BranchTargetFinder  branchTargetFinder,
                                       CodeAttributeEditor codeAttributeEditor)
    {
        this(patternConstants,
             patternInstructions,
             replacementConstants,
             replacementInstructions,
             branchTargetFinder,
             codeAttributeEditor,
             null);
    }


    /**
     * Creates a new InstructionSequenceReplacer.
     * @param patternConstants        any constants referenced by the pattern
     *                                instructions.
     * @param patternInstructions     the pattern instruction sequence.
     * @param replacementConstants    any constants referenced by the
     *                                replacement instructions.
     * @param replacementInstructions the replacement instruction sequence.
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
                                       Constant[]          replacementConstants,
                                       Instruction[]       replacementInstructions,
                                       BranchTargetFinder  branchTargetFinder,
                                       CodeAttributeEditor codeAttributeEditor,
                                       InstructionVisitor  extraInstructionVisitor)
    {
        this(new InstructionSequenceMatcher(patternConstants, patternInstructions),
             patternConstants,
             patternInstructions,
             replacementConstants,
             replacementInstructions,
             branchTargetFinder,
             codeAttributeEditor,
             extraInstructionVisitor);
    }



    /**
     * Creates a new InstructionSequenceReplacer.
     * @param instructionSequenceMatcher a suitable instruction sequence matcher.
     * @param patternConstants           any constants referenced by the pattern
     *                                   instructions.
     * @param patternInstructions        the pattern instruction sequence.
     * @param replacementConstants       any constants referenced by the
     *                                   replacement instructions.
     * @param replacementInstructions    the replacement instruction sequence.
     * @param branchTargetFinder         a branch target finder that has been
     *                                   initialized to indicate branch targets
     *                                   in the visited code.
     * @param codeAttributeEditor        a code editor that can be used for
     *                                   accumulating changes to the code.
     * @param extraInstructionVisitor    an optional extra visitor for all deleted
     *                                   load instructions.
     */
    protected InstructionSequenceReplacer(InstructionSequenceMatcher instructionSequenceMatcher,
                                          Constant[]                 patternConstants,
                                          Instruction[]              patternInstructions,
                                          Constant[]                 replacementConstants,
                                          Instruction[]              replacementInstructions,
                                          BranchTargetFinder         branchTargetFinder,
                                          CodeAttributeEditor        codeAttributeEditor,
                                          InstructionVisitor         extraInstructionVisitor)
    {
        this.instructionSequenceMatcher = instructionSequenceMatcher;
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
            int patternCount     = instructionSequenceMatcher.instructionCount();
            int replacementCount = replacementInstructions.length;

            if (DEBUG)
            {
                System.out.println("InstructionSequenceReplacer: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]");
                System.out.println("  Matched:");
                for (int index = 0; index < patternCount; index++)
                {
                    int matchedOffset = instructionSequenceMatcher.matchedInstructionOffset(index);
                    System.out.println("    "+InstructionFactory.create(codeAttribute.code, matchedOffset).toString(matchedOffset));
                }
                System.out.println("  Replacement:");
                for (int index = 0; index < replacementCount; index++)
                {
                    int matchedOffset = instructionSequenceMatcher.matchedInstructionOffset(Math.min(index, patternCount-1));
                    System.out.println("    " + replacementInstructionFactory.create(clazz, codeAttribute, index).shrink().toString(matchedOffset));
                }
            }

            // Is the replacement sequence shorter than the pattern sequence?
            if (replacementCount <= patternCount)
            {
                // Replace the instruction sequence.
                for (int index = 0; index < replacementCount; index++)
                {
                    codeAttributeEditor.replaceInstruction(instructionSequenceMatcher.matchedInstructionOffset(index),
                                                           replacementInstructionFactory.create(clazz, codeAttribute, index));
                }

                // Delete any remaining instructions in the matched sequence.
                for (int index = replacementCount; index < patternCount; index++)
                {
                    codeAttributeEditor.deleteInstruction(instructionSequenceMatcher.matchedInstructionOffset(index));
                }
            }
            else
            {
                // Replace the instruction sequence.
                for (int index = 0; index < patternCount; index++)
                {
                    codeAttributeEditor.replaceInstruction(instructionSequenceMatcher.matchedInstructionOffset(index),
                                                           replacementInstructionFactory.create(clazz, codeAttribute, index));
                }

                // Append the remaining instructions in the replacement
                // sequence.
                Instruction[] extraInstructions = new Instruction[replacementCount - patternCount];
                for (int index = 0; index < extraInstructions.length; index++)
                {
                    extraInstructions[index] = replacementInstructionFactory.create(clazz, codeAttribute, patternCount + index);
                }

                codeAttributeEditor.insertAfterInstruction(instructionSequenceMatcher.matchedInstructionOffset(patternCount - 1),
                                                           extraInstructions);
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
        public Instruction create(Clazz clazz, CodeAttribute codeAttribute, int index)
        {
            int matchedInstructionIndex =
                index < instructionSequenceMatcher.instructionCount() ?
                    index :
                    instructionSequenceMatcher.instructionCount() - 1;

            int matchedInstructionOffset =
                instructionSequenceMatcher.matchedInstructionOffset(matchedInstructionIndex);

            // Create the instruction.
            replacementInstructions[index].accept(clazz,
                                                  null,
                                                  codeAttribute,
                                                  matchedInstructionOffset,
                                                  this);

            // Return it.
            return replacementInstruction;
        }


        // Implementations for InstructionVisitor.

        public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
        {
            replacementInstruction =
                new SimpleInstruction(simpleInstruction.opcode,
                                      matchedArgument(clazz, method, codeAttribute, offset, simpleInstruction.constant));
        }


        public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
        {
            replacementInstruction =
                new VariableInstruction(variableInstruction.opcode,
                                        matchedArgument(clazz, method, codeAttribute, offset, variableInstruction.variableIndex),
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
                                      matchedBranchOffset(offset, branchInstruction.branchOffset));
        }


        public void visitTableSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, TableSwitchInstruction tableSwitchInstruction)
        {
            replacementInstruction =
                new TableSwitchInstruction(tableSwitchInstruction.opcode,
                                           matchedBranchOffset(offset, tableSwitchInstruction.defaultOffset),
                                           instructionSequenceMatcher.matchedArgument(tableSwitchInstruction.lowCase),
                                           instructionSequenceMatcher.matchedArgument(tableSwitchInstruction.highCase),
                                           matchedJumpOffsets(offset, tableSwitchInstruction.jumpOffsets));

        }


        public void visitLookUpSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LookUpSwitchInstruction lookUpSwitchInstruction)
        {
            replacementInstruction =
                new LookUpSwitchInstruction(lookUpSwitchInstruction.opcode,
                                            matchedBranchOffset(offset, lookUpSwitchInstruction.defaultOffset),
                                            instructionSequenceMatcher.matchedArguments(lookUpSwitchInstruction.cases),
                                            matchedJumpOffsets(offset, lookUpSwitchInstruction.jumpOffsets));
        }


        // Similar methods for pseudo-instructions.

        public void visitLabelInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Label label)
        {
            // Convert this pseudo-instruction into a corresponding
            // pseudo-instruction for the code attribute editor.
            // Then make sure we create a unique label, because
            // there may be other matching sequences.
            replacementInstruction =
                codeAttributeEditor.label(uniqueLabel(label.identifier));
        }


        public void visitCatchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Catch catch_)
        {
            // Convert this pseudo-instruction into a corresponding
            // pseudo-instruction for the code attribute editor.
            // Then make sure we create and reference unique labels,
            // because there may be other matching sequences.
            replacementInstruction =
                codeAttributeEditor.catch_(uniqueLabel(catch_.identifier),
                                           uniqueLabel(catch_.startOfffset),
                                           uniqueLabel(catch_.endOffset),
                                           matchedConstantIndex((ProgramClass)clazz,
                                                                catch_.catchType));
        }
    }


    /**
     * Returns the matched argument for the given pattern argument.
     */
    protected int matchedArgument(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int argument)
    {
        return matchedArgument(clazz, argument);
    }


    /**
     * Returns the matched argument for the given pattern argument.
     */
    protected int matchedArgument(Clazz clazz, int argument)
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
    protected int matchedConstantIndex(ProgramClass programClass, int constantIndex)
    {
        // Special case: do we have to create a concatenated string?
        if (constantIndex >= BOOLEAN_A_STRING &&
            constantIndex <= (STRING_A_STRING  | STRING_B_STRING))
        {
            // Create a new string constant and return its index.
            return new ConstantPoolEditor(programClass).addStringConstant(
                argumentAsString(programClass, constantIndex & 0xf, A) +
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


    /**
     * Returns the value of the specified matched branch offset.
     */
    protected int matchedBranchOffset(int offset, int branchOffset)
    {
        // Special case: is it a label?
        if (isLabel(branchOffset))
        {
            // Then make sure we reference a unique label, because
            // there may be other matching sequences.
            return uniqueLabel(branchOffset);
        }

        // Otherwise, just return the matched branch offset.
        return instructionSequenceMatcher.matchedBranchOffset(offset, branchOffset);
    }


    /**
     * Returns the values of the specified matched jump offsets.
     */
    protected int[] matchedJumpOffsets(int offset, int[] jumpOffsets)
    {
        // Special cases: are there any labels?
        for (int index = 0; index < jumpOffsets.length; index++)
        {
            if (isLabel(jumpOffsets[index]))
            {
                // Then make sure we reference a unique label, because
                // there may be other matching sequences.
                jumpOffsets[index] = uniqueLabel(jumpOffsets[index]);
            }
        }

        // Match any other jump offsets.
        return instructionSequenceMatcher.matchedJumpOffsets(offset, jumpOffsets);
    }


    private String argumentAsString(ProgramClass programClass,
                                    int          valueType,
                                    int          argument)
    {
        switch (valueType)
        {
            case BOOLEAN_STRING:
                return Boolean.toString((wasConstant(argument) ?
                    ((IntegerConstant)matchedConstant(programClass, argument)).getValue() :
                    matchedArgument(argument)) != 0);

            case CHAR_STRING:
                return Character.toString((char)(wasConstant(argument) ?
                    ((IntegerConstant)(matchedConstant(programClass, argument))).getValue() :
                    matchedArgument(argument)));

            case INT_STRING:
                return Integer.toString(wasConstant(argument) ?
                    ((IntegerConstant)(matchedConstant(programClass, argument))).getValue() :
                    matchedArgument(argument));

            case LONG_STRING:
                return Long.toString(wasConstant(argument) ?
                    ((LongConstant)(matchedConstant(programClass, argument))).getValue() :
                    matchedArgument(argument));

            case FLOAT_STRING:
                return Float.toString(wasConstant(argument) ?
                    ((FloatConstant)(matchedConstant(programClass, argument))).getValue() :
                    matchedArgument(argument));

            case DOUBLE_STRING:
                return Double.toString(wasConstant(argument) ?
                    ((DoubleConstant)(matchedConstant(programClass, argument))).getValue() :
                    matchedArgument(argument));

            case STRING_STRING:
                return
                    programClass.getStringString(instructionSequenceMatcher.matchedConstantIndex(argument));

            default:
                return "";
        }
    }


    protected InstructionSequenceMatcher getInstructionSequenceMatcher()
    {
        return instructionSequenceMatcher;
    }


    protected boolean wasConstant(int argument)
    {
        return instructionSequenceMatcher.wasConstant(argument);
    }


    protected Constant matchedConstant(ProgramClass programClass,
                                       int          argument)
    {
        return programClass.getConstant(instructionSequenceMatcher.matchedConstantIndex(argument));
    }


    protected int matchedArgument(int argument)
    {
        return instructionSequenceMatcher.matchedArgument(argument);
    }


    /**
     * Makes the given label identifier or offset unique for the current
     * matching instruction sequence.
     */
    private int uniqueLabel(int labelIdentifier)
    {
        return labelIdentifier |
               (instructionSequenceMatcher.matchedInstructionOffset(0) << 8);
    }


    // For convenience, we also define two pseudo-instructions, to conveniently
    // mark local labels and create new exceptions handlers.

    /**
     * Creates a new label that can be used as a pseudo-instruction to mark
     * a local offset. Its offset can be used as a branch target in
     * replacement instructions ({@link Label#offset()}).
     */
    public static Label label()
    {
        return new Label(labelCounter++);
    }


    /**
     * Creates a new catch instance that can be used as a pseudo-instruction
     * to mark the start of an exception handler. Its offset can be used as
     * a branch target in replacement instructions ({@link Label#offset()}).
     */
    public static Label catch_(int startOffset,
                               int endOffset,
                               int catchType)
    {
        return new Catch(labelCounter++,
                         startOffset,
                         endOffset,
                         catchType);
    }


    /**
     * Returns whether the given instruction offset actually represents a
     * label (which contains the actual offset).
     */
    private static boolean isLabel(int instructionOffset)
    {
        return (instructionOffset & 0xff000000) == LABEL_FLAG;
    }


    /**
     * This pseudo-instruction represents a label that marks an instruction
     * offset, for use in the context of the sequence replacer only.
     */
    public static class Label
    extends             Instruction
    {
        protected final int identifier;


        /**
         * Creates a new Label.
         * @param identifier an identifier that can be chosen freely.
         */
        private Label(int identifier)
        {
            this.identifier = identifier;
        }


        /**
         * Returns the offset that can then be used as a branch target in
         * other replacement instructions.
         */
        public int offset()
        {
            return LABEL_FLAG | identifier;
        }


        // Implementations for Instruction.

        public Instruction shrink()
        {
            return this;
        }


        public void write(byte[] code, int offset)
        {
        }


        protected void readInfo(byte[] code, int offset)
        {
            throw new UnsupportedOperationException("Can't read label instruction");
        }


        protected void writeInfo(byte[] code, int offset)
        {
            throw new UnsupportedOperationException("Can't write label instruction");
        }


        public int length(int offset)
        {
            return 0;
        }


        public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, InstructionVisitor instructionVisitor)
        {
            // Since this is not a standard instruction, it only works with
            // our own instruction visitor.
            MyReplacementInstructionFactory replacementInstructionFactory =
                (MyReplacementInstructionFactory)instructionVisitor;

            replacementInstructionFactory.visitLabelInstruction(clazz,
                                                                method,
                                                                codeAttribute,
                                                                offset,
                                                                this);
        }


        // Implementations for Object.

        public String toString()
        {
            return "label_"+offset();
        }
    }


    /**
     * This pseudo-instruction represents an exception handler,
     * for use in the context of the sequence replacer only.
     */
    private static class Catch
    extends              Label
    {
        private final int startOfffset;
        private final int endOffset;
        private final int catchType;


        /**
         * Creates a new Catch instance.
         * @param identifier  an identifier that can be chosen freely.
         * @param startOffset the start offset of the catch block.
         * @param endOffset   the end offset of the catch block.
         * @param catchType   the index of the catch type in the constant pool.
         */
        private Catch(int identifier,
                      int startOffset,
                      int endOffset,
                      int catchType)
        {
            super(identifier);

            this.startOfffset = startOffset;
            this.endOffset    = endOffset;
            this.catchType    = catchType;
        }


       // Implementations for Instruction.

        public Instruction shrink()
        {
            return this;
        }


        public void write(byte[] code, int offset)
        {
        }


        protected void readInfo(byte[] code, int offset)
        {
            throw new UnsupportedOperationException("Can't read catch instruction");
        }


        protected void writeInfo(byte[] code, int offset)
        {
            throw new UnsupportedOperationException("Can't write catch instruction");
        }


        public int length(int offset)
        {
            return super.length(offset);
        }


        public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, InstructionVisitor instructionVisitor)
        {
            // Since this is not a standard instruction, it only works with
            // our own instruction visitor.
            MyReplacementInstructionFactory replacementInstructionFactory =
                (MyReplacementInstructionFactory)instructionVisitor;

            replacementInstructionFactory.visitCatchInstruction(clazz,
                                                                method,
                                                                codeAttribute,
                                                                offset,
                                                                this);
        }


        // Implementations for Object.

        public String toString()
        {
            return "catch " +
                   (isLabel(startOfffset) ? "label_" : "") + startOfffset + ", " +
                   (isLabel(endOffset)    ? "label_" : "") + endOffset    + ", #" +
                   catchType;
        }
    }
}
