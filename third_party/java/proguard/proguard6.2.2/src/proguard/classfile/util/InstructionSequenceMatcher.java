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
package proguard.classfile.util;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;

import java.util.Arrays;

/**
 * This InstructionVisitor checks whether a given pattern instruction sequence
 * occurs in the instructions that are visited. The arguments of the
 * instruction sequence can be wildcards that are matched.
 *
 * @author Eric Lafortune
 */
public class InstructionSequenceMatcher
extends      SimplifiedVisitor
implements   InstructionVisitor,
             ConstantVisitor
{
    //*
    private static final boolean DEBUG      = false;
    private static final boolean DEBUG_MORE = false;
    /*/
    public  static       boolean DEBUG      = System.getProperty("ism")  != null;
    public  static       boolean DEBUG_MORE = System.getProperty("ismm") != null;
    //*/

    public static final int X = 0x40000000;
    public static final int Y = 0x40000001;
    public static final int Z = 0x40000002;

    public static final int A = 0x40000003;
    public static final int B = 0x40000004;
    public static final int C = 0x40000005;
    public static final int D = 0x40000006;
    public static final int E = 0x40000007;
    public static final int F = 0x40000008;
    public static final int G = 0x40000009;
    public static final int H = 0x4000000a;
    public static final int I = 0x4000000b;
    public static final int J = 0x4000000c;
    public static final int K = 0x4000000d;
    public static final int L = 0x4000000e;
    public static final int M = 0x4000000f;
    public static final int N = 0x40000010;
    public static final int O = 0x40000011;
    public static final int P = 0x40000012;
    public static final int Q = 0x40000013;
    public static final int R = 0x40000014;


    protected final Constant[]    patternConstants;
    protected final Instruction[] patternInstructions;

    private boolean      matching;
    private int          patternInstructionIndex;
    private final int[]  matchedInstructionOffsets;
    private int          matchedArgumentFlags;
    private final int[]  matchedArguments = new int[21];
    private final long[] matchedConstantFlags;
    private final int[]  matchedConstantIndices;
    private int          constantFlags;
    private int          previousConstantFlags;

    // Fields acting as a parameter and a return value for visitor methods.
    protected Constant patternConstant;
    protected boolean  matchingConstant;


    /**
     * Creates a new InstructionSequenceMatcher.
     * @param patternConstants        any constants referenced by the pattern
     *                                instruction.
     * @param patternInstructions     the pattern instruction sequence.
     */
    public InstructionSequenceMatcher(Constant[]    patternConstants,
                                      Instruction[] patternInstructions)
    {
        this.patternConstants    = patternConstants;
        this.patternInstructions = patternInstructions;

        matchedInstructionOffsets = new int[patternInstructions.length];
        matchedConstantFlags      = new long[(patternConstants.length + 63) / 64];
        matchedConstantIndices    = new int[patternConstants.length];
    }


    /**
     * Starts matching from the first instruction again next time.
     */
    public void reset()
    {
        patternInstructionIndex = 0;
        matchedArgumentFlags    = 0;

        Arrays.fill(matchedConstantFlags, 0L);

        previousConstantFlags = constantFlags;
        constantFlags         = 0;
    }


    /**
     * Returns whether the complete pattern sequence has been matched.
     */
    public boolean isMatching()
    {
        return matching;
    }


    /**
     * Returns the number of instructions in the pattern sequence.
     */
    public int instructionCount()
    {
        return patternInstructions.length;
    }


    /**
     * Returns the matched instruction offset of the specified pattern
     * instruction.
     */
    public int matchedInstructionOffset(int index)
    {
        return matchedInstructionOffsets[index];
    }


    /**
     * Returns whether the specified wildcard argument was a constant from
     * the constant pool in the most recent match.
     */
    public boolean wasConstant(int argument)
    {
        return (previousConstantFlags & (1 << (argument - X))) != 0;
    }


    /**
     * Returns the value of the specified matched argument (wildcard or not).
     */
    public int matchedArgument(int argument)
    {
        int argumentIndex = argument - X;
        return argumentIndex < 0 ?
            argument :
            matchedArguments[argumentIndex];
    }


    /**
     * Returns the values of the specified matched arguments (wildcard or not).
     */
    public int[] matchedArguments(int[] arguments)
    {
        int[] matchedArguments = new int[arguments.length];

        for (int index = 0; index < arguments.length; index++)
        {
            matchedArguments[index] = matchedArgument(arguments[index]);
        }

        return matchedArguments;
    }


    /**
     * Returns the index of the specified matched constant (wildcard or not).
     */
    public int matchedConstantIndex(int constantIndex)
    {
        int argumentIndex = constantIndex - X;
        return argumentIndex < 0 ?
            matchedConstantIndices[constantIndex] :
            matchedArguments[argumentIndex];
    }


    /**
     * Returns the value of the specified matched branch offset (wildcard or
     * not).
     */
    public int matchedBranchOffset(int offset, int branchOffset)
    {
        int argumentIndex = branchOffset - X;
        return argumentIndex < 0 ?
            branchOffset :
            matchedArguments[argumentIndex] - offset;
    }


    /**
     * Returns the values of the specified matched jump offsets (wildcard or
     * not).
     */
    public int[] matchedJumpOffsets(int offset, int[] jumpOffsets)
    {
        int[] matchedJumpOffsets = new int[jumpOffsets.length];

        for (int index = 0; index < jumpOffsets.length; index++)
        {
            matchedJumpOffsets[index] = matchedBranchOffset(offset,
                                                            jumpOffsets[index]);
        }

        return matchedJumpOffsets;
    }


    // Implementations for InstructionVisitor.

    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        Instruction patternInstruction = patternInstructions[patternInstructionIndex];

        // Check if the instruction matches the next instruction in the sequence.
        boolean condition =
            matchingOpcodes(simpleInstruction, patternInstruction) &&
            matchingArguments(simpleInstruction.constant,
                              ((SimpleInstruction)patternInstruction).constant);

        // Check if the instruction sequence is matching now.
        checkMatch(condition,
                   clazz,
                   method,
                   codeAttribute,
                   offset,
                   simpleInstruction);
    }


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        Instruction patternInstruction = patternInstructions[patternInstructionIndex];

        // Check if the instruction matches the next instruction in the sequence.
        boolean condition =
            matchingOpcodes(variableInstruction, patternInstruction) &&
            matchingArguments(variableInstruction.variableIndex,
                              ((VariableInstruction)patternInstruction).variableIndex) &&
            matchingArguments(variableInstruction.constant,
                              ((VariableInstruction)patternInstruction).constant);

        // Check if the instruction sequence is matching now.
        checkMatch(condition,
                   clazz,
                   method,
                   codeAttribute,
                   offset,
                   variableInstruction);
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        Instruction patternInstruction = patternInstructions[patternInstructionIndex];

        // Check if the instruction matches the next instruction in the sequence.
        boolean condition =
            matchingOpcodes(constantInstruction, patternInstruction) &&
            matchingConstantIndices(clazz,
                                    constantInstruction.constantIndex,
                                    ((ConstantInstruction)patternInstruction).constantIndex) &&
            matchingArguments(constantInstruction.constant,
                              ((ConstantInstruction)patternInstruction).constant);

        // Check if the instruction sequence is matching now.
        checkMatch(condition,
                   clazz,
                   method,
                   codeAttribute,
                   offset,
                   constantInstruction);
    }


    public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
    {
        Instruction patternInstruction = patternInstructions[patternInstructionIndex];

        // Check if the instruction matches the next instruction in the from
        // sequence.
        boolean condition =
            matchingOpcodes(branchInstruction, patternInstruction) &&
            matchingBranchOffsets(offset,
                                  branchInstruction.branchOffset,
                                  ((BranchInstruction)patternInstruction).branchOffset);

        // Check if the instruction sequence is matching now.
        checkMatch(condition,
                   clazz,
                   method,
                   codeAttribute,
                   offset,
                   branchInstruction);
    }


    public void visitTableSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, TableSwitchInstruction tableSwitchInstruction)
    {
        Instruction patternInstruction = patternInstructions[patternInstructionIndex];

        // Check if the instruction matches the next instruction in the sequence.
        boolean condition =
            matchingOpcodes(tableSwitchInstruction, patternInstruction) &&
            matchingBranchOffsets(offset,
                                  tableSwitchInstruction.defaultOffset,
                                  ((TableSwitchInstruction)patternInstruction).defaultOffset) &&
            matchingArguments(tableSwitchInstruction.lowCase,
                              ((TableSwitchInstruction)patternInstruction).lowCase)  &&
            matchingArguments(tableSwitchInstruction.highCase,
                              ((TableSwitchInstruction)patternInstruction).highCase) &&
            matchingJumpOffsets(offset,
                                tableSwitchInstruction.jumpOffsets,
                                ((TableSwitchInstruction)patternInstruction).jumpOffsets);

        // Check if the instruction sequence is matching now.
        checkMatch(condition,
                   clazz,
                   method,
                   codeAttribute,
                   offset,
                   tableSwitchInstruction);
    }


    public void visitLookUpSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LookUpSwitchInstruction lookUpSwitchInstruction)
    {
        Instruction patternInstruction = patternInstructions[patternInstructionIndex];

        // Check if the instruction matches the next instruction in the sequence.
        boolean condition =
            matchingOpcodes(lookUpSwitchInstruction, patternInstruction) &&
            matchingBranchOffsets(offset,
                                  lookUpSwitchInstruction.defaultOffset,
                                  ((LookUpSwitchInstruction)patternInstruction).defaultOffset) &&
            matchingArguments(lookUpSwitchInstruction.cases,
                              ((LookUpSwitchInstruction)patternInstruction).cases) &&
            matchingJumpOffsets(offset,
                                lookUpSwitchInstruction.jumpOffsets,
                                ((LookUpSwitchInstruction)patternInstruction).jumpOffsets);

        // Check if the instruction sequence is matching now.
        checkMatch(condition,
                   clazz,
                   method,
                   codeAttribute,
                   offset,
                   lookUpSwitchInstruction);
    }


    // Implementations for ConstantVisitor.

    public void visitIntegerConstant(Clazz clazz, IntegerConstant integerConstant)
    {
        IntegerConstant integerPatternConstant = (IntegerConstant)patternConstant;

        // Compare the integer values.
        matchingConstant = integerConstant.getValue() ==
                           integerPatternConstant.getValue();
    }


    public void visitLongConstant(Clazz clazz, LongConstant longConstant)
    {
        LongConstant longPatternConstant = (LongConstant)patternConstant;

        // Compare the long values.
        matchingConstant = longConstant.getValue() ==
                           longPatternConstant.getValue();
    }


    public void visitFloatConstant(Clazz clazz, FloatConstant floatConstant)
    {
        FloatConstant floatPatternConstant = (FloatConstant)patternConstant;

        // Compare the float values.
        matchingConstant = floatConstant.getValue() ==
                           floatPatternConstant.getValue();
    }


    public void visitDoubleConstant(Clazz clazz, DoubleConstant doubleConstant)
    {
        DoubleConstant doublePatternConstant = (DoubleConstant)patternConstant;

        // Compare the double values.
        matchingConstant = doubleConstant.getValue() ==
                           doublePatternConstant.getValue();
    }


    public void visitPrimitiveArrayConstant(Clazz clazz, PrimitiveArrayConstant primitiveArrayConstant)
    {
        //PrimitiveArrayConstant primitiveArrayPatternConstant = (PrimitiveArrayConstant)patternConstant;
        //
        //// Compare the primitive array values.
        //matchingConstant =
        //    primitiveArrayConstant.getLength() == primitiveArrayPatternConstant.getLength() &&
        //    ArrayUtil.equal(primitiveArrayConstant.getValues(),
        //                    primitiveArrayPatternConstant.getValues(),
        //                    primitiveArrayPatternConstant.getLength());
        throw new UnsupportedOperationException();
    }


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        StringConstant stringPatternConstant = (StringConstant)patternConstant;

        // Check the UTF-8 constant.
        matchingConstant =
            matchingConstantIndices(clazz,
                                    stringConstant.u2stringIndex,
                                    stringPatternConstant.u2stringIndex);
    }


    public void visitUtf8Constant(Clazz clazz, Utf8Constant utf8Constant)
    {
        Utf8Constant utf8PatternConstant = (Utf8Constant)patternConstant;

        // Compare the actual strings.
        matchingConstant = utf8Constant.getString().equals(
                           utf8PatternConstant.getString());
    }


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        DynamicConstant dynamicPatternConstant = (DynamicConstant)patternConstant;

        // Check the bootstrap method and the name and type.
        matchingConstant =
            matchingConstantIndices(clazz,
                                    dynamicConstant.getBootstrapMethodAttributeIndex(),
                                    dynamicPatternConstant.getBootstrapMethodAttributeIndex()) &&
            matchingConstantIndices(clazz,
                                    dynamicConstant.getNameAndTypeIndex(),
                                    dynamicPatternConstant.getNameAndTypeIndex());
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        InvokeDynamicConstant invokeDynamicPatternConstant = (InvokeDynamicConstant)patternConstant;

        // Check the bootstrap method and the name and type.
        matchingConstant =
            matchingConstantIndices(clazz,
                                    invokeDynamicConstant.getBootstrapMethodAttributeIndex(),
                                    invokeDynamicPatternConstant.getBootstrapMethodAttributeIndex()) &&
            matchingConstantIndices(clazz,
                                    invokeDynamicConstant.getNameAndTypeIndex(),
                                    invokeDynamicPatternConstant.getNameAndTypeIndex());
    }


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        MethodHandleConstant methodHandlePatternConstant = (MethodHandleConstant)patternConstant;

        // Check the handle type and the name and type.
        matchingConstant =
            matchingArguments(methodHandleConstant.getReferenceKind(),
                              methodHandlePatternConstant.getReferenceKind()) &&
            matchingConstantIndices(clazz,
                                    methodHandleConstant.getReferenceIndex(),
                                    methodHandlePatternConstant.getReferenceIndex());
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        RefConstant refPatternConstant = (RefConstant)patternConstant;

        // Check the class and the name and type.
        matchingConstant =
            matchingConstantIndices(clazz,
                                    refConstant.getClassIndex(),
                                    refPatternConstant.getClassIndex()) &&
            matchingConstantIndices(clazz,
                                    refConstant.getNameAndTypeIndex(),
                                    refPatternConstant.getNameAndTypeIndex());
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        ClassConstant classPatternConstant = (ClassConstant)patternConstant;

        // Check the class name.
        matchingConstant =
            matchingConstantIndices(clazz,
                                    classConstant.u2nameIndex,
                                    classPatternConstant.u2nameIndex);
    }


    public void visitMethodTypeConstant(Clazz clazz, MethodTypeConstant methodTypeConstant)
    {
        MethodTypeConstant typePatternConstant = (MethodTypeConstant)patternConstant;

        // Check the descriptor.
        matchingConstant =
            matchingConstantIndices(clazz,
                                    methodTypeConstant.u2descriptorIndex,
                                    typePatternConstant.u2descriptorIndex);
    }


    public void visitNameAndTypeConstant(Clazz clazz, NameAndTypeConstant nameAndTypeConstant)
    {
        NameAndTypeConstant typePatternConstant = (NameAndTypeConstant)patternConstant;

        // Check the name and the descriptor.
        matchingConstant =
            matchingConstantIndices(clazz,
                                    nameAndTypeConstant.u2nameIndex,
                                    typePatternConstant.u2nameIndex) &&
            matchingConstantIndices(clazz,
                                    nameAndTypeConstant.u2descriptorIndex,
                                    typePatternConstant.u2descriptorIndex);
    }


    // Small utility methods.

    protected boolean matchingOpcodes(Instruction instruction1,
                                      Instruction instruction2)
    {
        // Check the opcode.
        return instruction1.opcode            == instruction2.opcode ||
               instruction1.canonicalOpcode() == instruction2.opcode;
    }


    protected boolean matchingArguments(int argument1,
                                        int argument2)
    {
        int argumentIndex = argument2 - X;
        if (argumentIndex < 0)
        {
            // Check the literal argument.
            return argument1 == argument2;
        }
        else if (!isMatchingArgumentIndex(argumentIndex))
        {
            // Store the wildcard argument.
            setMatchingArgument(argumentIndex, argument1);

            return true;
        }
        else
        {
            // Check the previously stored wildcard argument.
            return matchedArguments[argumentIndex] == argument1;
        }
    }


    /**
     * Marks the specified argument (by index) as matching the specified
     * argument value.
     */
    private void setMatchingArgument(int argumentIndex,
                                     int argument)
    {
        matchedArguments[argumentIndex] = argument;
        matchedArgumentFlags |= 1 << argumentIndex;
    }


    /**
     * Returns whether the specified wildcard argument (by index) has been
     * matched.
     */
    private boolean isMatchingArgumentIndex(int argumentIndex)
    {
        return (matchedArgumentFlags & (1 << argumentIndex)) != 0;
    }


    protected boolean matchingArguments(int[] arguments1,
                                        int[] arguments2)
    {
        if (arguments1.length != arguments2.length)
        {
            return false;
        }

        for (int index = 0; index < arguments1.length; index++)
        {
            if (!matchingArguments(arguments1[index], arguments2[index]))
            {
                return false;
            }
        }

        return true;
    }


    protected boolean matchingConstantIndices(Clazz clazz,
                                              int   constantIndex1,
                                              int   constantIndex2)
    {
        if (constantIndex2 >= X)
        {
            // Remember that we are trying to match a constant.
            constantFlags |= 1 << (constantIndex2 - X);

            // Check the constant index.
            return matchingArguments(constantIndex1, constantIndex2);
        }
        else if (!isMatchingConstantIndex(constantIndex2))
        {
            // Check the actual constant.
            matchingConstant = false;
            patternConstant  = patternConstants[constantIndex2];

            if (clazz.getTag(constantIndex1) == patternConstant.getTag())
            {
                clazz.constantPoolEntryAccept(constantIndex1, this);

                if (matchingConstant)
                {
                    // Store the constant index.
                    setMatchingConstant(constantIndex2, constantIndex1);
                }
            }

            return matchingConstant;
        }
        else
        {
            // Check a previously stored constant index.
            return matchedConstantIndices[constantIndex2] == constantIndex1;
        }
    }


    /**
     * Marks the specified constant (by index) as matching the specified
     * constant index value.
     */
    private void setMatchingConstant(int constantIndex,
                                     int constantIndex1)
    {
        matchedConstantIndices[constantIndex] = constantIndex1;
        matchedConstantFlags[constantIndex / 64] |= 1L << constantIndex;
    }


    /**
     * Returns whether the specified wildcard constant has been matched.
     */
    private boolean isMatchingConstantIndex(int constantIndex)
    {
        return (matchedConstantFlags[constantIndex / 64] & (1L << constantIndex)) != 0;
    }


    protected boolean matchingBranchOffsets(int offset,
                                            int branchOffset1,
                                            int branchOffset2)
    {
        int argumentIndex = branchOffset2 - X;
        if (argumentIndex < 0)
        {
            // Check the literal argument.
            return branchOffset1 == branchOffset2;
        }
        else if (!isMatchingArgumentIndex(argumentIndex))
        {
            // Store a wildcard argument.
            setMatchingArgument(argumentIndex, offset + branchOffset1);

            return true;
        }
        else
        {
            // Check the previously stored wildcard argument.
            return matchedArguments[argumentIndex] == offset + branchOffset1;
        }
    }


    protected boolean matchingJumpOffsets(int   offset,
                                          int[] jumpOffsets1,
                                          int[] jumpOffsets2)
    {
        if (jumpOffsets1.length != jumpOffsets2.length)
        {
            return false;
        }

        for (int index = 0; index < jumpOffsets1.length; index++)
        {
            if (!matchingBranchOffsets(offset,
                                       jumpOffsets1[index],
                                       jumpOffsets2[index]))
            {
                return false;
            }
        }

        return true;
    }


    private void checkMatch(boolean       condition,
                            Clazz         clazz,
                            Method        method,
                            CodeAttribute codeAttribute,
                            int           offset,
                            Instruction   instruction)
    {
        if (DEBUG_MORE)
        {
            System.out.println("InstructionSequenceMatcher: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]: "+instruction.toString(offset)+(condition?"\t== ":"\t   ")+patternInstructions[patternInstructionIndex].toString(patternInstructionIndex));
        }

        // Did the instruction match?
        if (condition)
        {
            // Remember the offset of the matching instruction.
            matchedInstructionOffsets[patternInstructionIndex] = offset;

            // Try to match the next instruction next time.
            patternInstructionIndex++;

            // Did we match all instructions in the sequence?
            matching = patternInstructionIndex == patternInstructions.length;

            if (matching)
            {
                // Allow subclasses to perform a final check on additional constraints.
                matching &= finalMatch(clazz, method, codeAttribute, offset, instruction);

                if (DEBUG)
                {
                    System.out.println("InstructionSequenceMatcher: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]");
                    for (int index = 0; index < patternInstructionIndex; index++)
                    {
                        System.out.println("    "+InstructionFactory.create(codeAttribute.code, matchedInstructionOffsets[index]).toString(matchedInstructionOffsets[index]));
                    }

                    for (int index = 0; index < matchedArguments.length; index++)
                    {
                        if ((matchedArgumentFlags & (1 << index)) != 0)
                        {
                            System.out.println("      Arg #"+index+": "+matchedArguments[index]);
                        }
                    }

                    for (int index = 0; index < matchedConstantIndices.length; index++)
                    {
                        if (isMatchingConstantIndex(index))
                        {
                            System.out.println("      Constant #"+index+": "+matchedConstantIndices[index]);
                        }
                    }
                }

                // Start matching from the first instruction again next time.
                reset();
            }
        }
        else
        {
            // The instruction didn't match.
            matching = false;

            // Is this a failed second instruction?
            boolean retry = patternInstructionIndex == 1;

            // Start matching from the first instruction next time.
            reset();

            // Retry a failed second instruction as a first instruction.
            if (retry)
            {
                instruction.accept(clazz, method, codeAttribute, offset, this);
            }
        }
    }


    /**
     * Performs a final check on the candidate sequence to match,
     * after the pattern has been successfully fully matched with the
     * sequence. Subclasses may override this method to implement
     * additional constraints on the matched sequences.
     *
     * @param clazz
     * @param method
     * @param codeAttribute
     * @param offset
     * @param instruction
     * @return
     */
    protected boolean finalMatch(Clazz         clazz,
                                 Method        method,
                                 CodeAttribute codeAttribute,
                                 int           offset,
                                 Instruction   instruction   )
    {
        return true;
    }
}
