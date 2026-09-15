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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.annotation.target.*;
import proguard.classfile.attribute.annotation.target.visitor.*;
import proguard.classfile.attribute.annotation.visitor.TypeAnnotationVisitor;
import proguard.classfile.attribute.preverification.*;
import proguard.classfile.attribute.preverification.visitor.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassPrinter;
import proguard.util.ArrayUtil;

import java.util.Arrays;

/**
 * This AttributeVisitor accumulates instructions and exceptions, and then
 * copies them into code attributes that it visits.
 *
 * @author Eric Lafortune
 */
public class CodeAttributeComposer
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor,
             ExceptionInfoVisitor,
             StackMapFrameVisitor,
             VerificationTypeVisitor,
             LineNumberInfoVisitor,
             LocalVariableInfoVisitor,
             LocalVariableTypeInfoVisitor,
             TypeAnnotationVisitor,
             TargetInfoVisitor,
             LocalVariableTargetElementVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    public  static       boolean DEBUG = false;
    //*/


    private static final int MAXIMUM_LEVELS = 32;
    private static final int INVALID        = -1;


    private final boolean allowExternalBranchTargets;
    private final boolean allowExternalExceptionHandlers;
    private final boolean shrinkInstructions;

    private int maximumCodeLength;
    private int codeLength;
    private int exceptionTableLength;
    private int lineNumberTableLength;
    private int level = -1;

    private byte[]  code                  = new byte[ClassConstants.TYPICAL_CODE_LENGTH];
    private int[]   oldInstructionOffsets = new int[ClassConstants.TYPICAL_CODE_LENGTH];

    private final int[]   codeFragmentOffsets  = new int[MAXIMUM_LEVELS];
    private final int[]   codeFragmentLengths  = new int[MAXIMUM_LEVELS];
    private final int[][] instructionOffsetMap = new int[MAXIMUM_LEVELS][ClassConstants.TYPICAL_CODE_LENGTH + 1];

    private ExceptionInfo[]  exceptionTable  = new ExceptionInfo[ClassConstants.TYPICAL_EXCEPTION_TABLE_LENGTH];
    private LineNumberInfo[] lineNumberTable = new LineNumberInfo[ClassConstants.TYPICAL_LINE_NUMBER_TABLE_LENGTH];

    private int expectedStackMapFrameOffset;

    private final StackSizeUpdater    stackSizeUpdater    = new StackSizeUpdater();
    private final VariableSizeUpdater variableSizeUpdater = new VariableSizeUpdater();
    private final InstructionWriter   instructionWriter   = new InstructionWriter();


    /**
     * Creates a new CodeAttributeComposer that doesn't allow external branch
     * targets or exception handlers and that automatically shrinks
     * instructions.
     */
    public CodeAttributeComposer()
    {
        this(false, false, true);
    }


    /**
     * Creates a new CodeAttributeComposer.
     * @param allowExternalBranchTargets     specifies whether branch targets
     *                                       can lie outside the code fragment
     *                                       of the branch instructions.
     * @param allowExternalExceptionHandlers specifies whether exception
     *                                       handlers can lie outside the code
     *                                       fragment in which exceptions are
     *                                       defined.
     * @param shrinkInstructions             specifies whether instructions
     *                                       should automatically be shrunk
     *                                       before being written.
     */
    public CodeAttributeComposer(boolean allowExternalBranchTargets,
                                 boolean allowExternalExceptionHandlers,
                                 boolean shrinkInstructions)
    {
        this.allowExternalBranchTargets     = allowExternalBranchTargets;
        this.allowExternalExceptionHandlers = allowExternalExceptionHandlers;
        this.shrinkInstructions             = shrinkInstructions;
    }


    /**
     * Starts a new code definition.
     */
    public void reset()
    {
        maximumCodeLength     = 0;
        codeLength            = 0;
        exceptionTableLength  = 0;
        lineNumberTableLength = 0;
        level                 = -1;

        // Make sure the instruction writer has at least the same buffer size
        // as the local arrays.
        instructionWriter.reset(code.length);
    }


    /**
     * Starts a new code fragment. Branch instructions that are added are
     * assumed to be relative within such code fragments.
     * @param maximumCodeFragmentLength the maximum length of the code that will
     *                                  be added as part of this fragment (more
     *                                  precisely, the maximum old instruction
     *                                  offset or label that is specified, plus
     *                                  one).
     */
    public void beginCodeFragment(int maximumCodeFragmentLength)
    {
        level++;

        if (level >= MAXIMUM_LEVELS)
        {
            throw new IllegalArgumentException("Maximum number of code fragment levels exceeded ["+level+"]");
        }

        // Make sure there is sufficient space for adding the code fragment.
        // It's only a rough initial estimate for the code length, not even
        // necessarily a length expressed in bytes.
        maximumCodeLength += maximumCodeFragmentLength;

        ensureCodeLength(maximumCodeLength);

        // Try to reuse the previous array for this code fragment.
        if (instructionOffsetMap[level].length <= maximumCodeFragmentLength)
        {
            instructionOffsetMap[level] = new int[maximumCodeFragmentLength + 1];
        }

        // Initialize the offset map.
        for (int index = 0; index <= maximumCodeFragmentLength; index++)
        {
            instructionOffsetMap[level][index] = INVALID;
        }

        // Remember the location of the code fragment.
        codeFragmentOffsets[level] = codeLength;
        codeFragmentLengths[level] = maximumCodeFragmentLength;
    }


    /**
     * Appends the given instruction with the given old offset.
     * Branch instructions must fit, for instance by enabling automatic
     * shrinking of instructions.
     * @param oldInstructionOffset the old offset of the instruction, to which
     *                             branches and other references in the current
     *                             code fragment are pointing.
     * @param instruction          the instruction to be appended.
     */
    public void appendInstruction(int         oldInstructionOffset,
                                  Instruction instruction)
    {
        if (shrinkInstructions)
        {
            instruction = instruction.shrink();
        }

        if (DEBUG)
        {
            println("["+codeLength+"] <- ", instruction.toString(oldInstructionOffset));
        }

        // Make sure the code and offset arrays are large enough.
        int newCodeLength = codeLength + instruction.length(codeLength);

        ensureCodeLength(newCodeLength);

        // Remember the old offset of the appended instruction.
        oldInstructionOffsets[codeLength] = oldInstructionOffset;

        // Fill out the new offset of the appended instruction.
        instructionOffsetMap[level][oldInstructionOffset] = codeLength;

        // Write the instruction. The instruction writer may widen it later on,
        // if necessary.
        instruction.accept(null,
                           null,
                           new CodeAttribute(0, 0, 0, 0, code, 0, null, 0, null),
                           codeLength,
                           instructionWriter);
        //instruction.write(code, codeLength);

        // Continue appending at the next instruction offset.
        codeLength = newCodeLength;
    }


    /**
     * Appends the given label with the given old offset.
     * @param oldInstructionOffset the old offset of the label, to which
     *                             branches and other references in the current
     *                             code fragment are pointing.
     */
    public void appendLabel(int oldInstructionOffset)
    {
        if (DEBUG)
        {
            println("["+codeLength+"] <- ", "[" + oldInstructionOffset + "] (label)");
        }

        // Make sure the code and offset arrays are large enough.
        ensureCodeLength(codeLength + 1);

        // Remember the old offset of the following instruction.
        oldInstructionOffsets[codeLength] = oldInstructionOffset;

        // Fill out the new offset of the following instruction.
        instructionOffsetMap[level][oldInstructionOffset] = codeLength;
    }


    /**
     * Appends the given instruction without defined offsets.
     * @param instructions the instructions to be appended.
     */
    public void appendInstructions(Instruction[] instructions)
    {
        for (int index = 0; index < instructions.length; index++)
        {
            appendInstruction(instructions[index]);
        }
    }


    /**
     * Appends the given instruction without a defined offset.
     * Branch instructions should have a label, to allow computing the
     * new relative offset.
     * Branch instructions must fit, for instance by enabling automatic
     * shrinking of instructions.
     * @param instruction the instruction to be appended.
     */
    public void appendInstruction(Instruction instruction)
    {
        if (shrinkInstructions)
        {
            instruction = instruction.shrink();
        }

        if (DEBUG)
        {
            println("["+codeLength+"] <- ", instruction.toString());
        }

        // Make sure the code array is large enough.
        int newCodeLength = codeLength + instruction.length(codeLength);

        ensureCodeLength(newCodeLength);

        // Write the instruction. The instruction writer may widen it later on,
        // if necessary.
        instruction.accept(null,
                           null,
                           new CodeAttribute(0, 0, 0, 0, code, 0, null, 0, null),
                           codeLength,
                           instructionWriter);
        //instruction.write(code, codeLength);

        // Continue appending at the next instruction offset.
        codeLength = newCodeLength;
    }


    /**
     * Appends the given exception to the exception table.
     * @param exceptionInfo the exception to be appended.
     */
    public void appendException(ExceptionInfo exceptionInfo)
    {
        if (DEBUG)
        {
            print("         ", "Exception ["+exceptionInfo.u2startPC+" -> "+exceptionInfo.u2endPC+": "+exceptionInfo.u2handlerPC+"]");
        }

        // Remap the exception right away.
        visitExceptionInfo(null, null, null, exceptionInfo);

        if (DEBUG)
        {
            System.out.println(" -> ["+exceptionInfo.u2startPC+" -> "+exceptionInfo.u2endPC+": "+exceptionInfo.u2handlerPC+"]");
        }

        // Don't add the exception if its instruction range is empty.
        if (exceptionInfo.u2startPC == exceptionInfo.u2endPC)
        {
            if (DEBUG)
            {
                println("         ", "  (not added because of empty instruction range)");
            }

            return;
        }

        // Add the exception.
        exceptionTable =
            (ExceptionInfo[])ArrayUtil.add(exceptionTable,
                                           exceptionTableLength++,
                                           exceptionInfo);
    }


    /**
     * Inserts the given line number at the appropriate position in the line
     * number table.
     * @param lineNumberInfo the line number to be inserted.
     * @return the index where the line number was actually inserted.
     */
    public int insertLineNumber(LineNumberInfo lineNumberInfo)
    {
        return insertLineNumber(0, lineNumberInfo);
    }


    /**
     * Inserts the given line number at the appropriate position in the line
     * number table.
     * @param minimumIndex   the minimum index where the line number may be
     *                       inserted.
     * @param lineNumberInfo the line number to be inserted.
     * @return the index where the line number was inserted.
     */
    public int insertLineNumber(int minimumIndex, LineNumberInfo lineNumberInfo)
    {
        if (DEBUG)
        {
            print("         ", "Line number ["+lineNumberInfo.u2startPC+"]");
        }

        // Remap the line number right away.
        visitLineNumberInfo(null, null, null, lineNumberInfo);

        if (DEBUG)
        {
            System.out.println(" -> ["+lineNumberInfo.u2startPC+"] line "+lineNumberInfo.u2lineNumber+(lineNumberInfo.getSource()==null ? "":" ["+lineNumberInfo.getSource()+"]"));
        }

        lineNumberTable =
            (LineNumberInfo[])ArrayUtil.extendArray(lineNumberTable,
                                                    lineNumberTableLength + 1);

        // Find the insertion index, starting from the end.
        // Don't insert before a negative line number, in case of a tie.
        int index = lineNumberTableLength++;
        while (index > minimumIndex &&
               (lineNumberTable[index - 1].u2startPC    >  lineNumberInfo.u2startPC ||
                lineNumberTable[index - 1].u2startPC    >= lineNumberInfo.u2startPC &&
                lineNumberTable[index - 1].u2lineNumber >= 0))
        {
            lineNumberTable[index] = lineNumberTable[--index];
        }

        lineNumberTable[index] = lineNumberInfo;

        return index;
    }


    /**
     * Appends the given line number to the line number table.
     * @param lineNumberInfo the line number to be appended.
     */
    public void appendLineNumber(LineNumberInfo lineNumberInfo)
    {
        if (DEBUG)
        {
            print("         ", "Line number ["+lineNumberInfo.u2startPC+"]");
        }

        // Remap the line number right away.
        visitLineNumberInfo(null, null, null, lineNumberInfo);

        if (DEBUG)
        {
            System.out.println(" -> ["+lineNumberInfo.u2startPC+"] line "+lineNumberInfo.u2lineNumber+(lineNumberInfo.getSource()==null ? "":" ["+lineNumberInfo.getSource()+"]"));
        }

        // Add the line number.
        lineNumberTable =
            (LineNumberInfo[])ArrayUtil.add(lineNumberTable,
                                            lineNumberTableLength++,
                                            lineNumberInfo);
    }


    /**
     * Wraps up the current code fragment, continuing with the previous one on
     * the stack.
     */
    public void endCodeFragment()
    {
        if (level < 0)
        {
            throw new IllegalArgumentException("Code fragment not begun ["+level+"]");
        }

        // Remap the instructions of the code fragment.
        int instructionOffset = codeFragmentOffsets[level];
        while (instructionOffset < codeLength)
        {
            // Get the next instruction.
            Instruction instruction = InstructionFactory.create(code, instructionOffset);

            // Does this instruction still have to be remapped?
            if (oldInstructionOffsets[instructionOffset] >= 0)
            {
                // Adapt the instruction for its new offset.
                instruction.accept(null, null, null, instructionOffset, this);

                // Write the instruction back. The instruction writer may still
                // widen it later on, if necessary.
                instruction.accept(null,
                                   null,
                                   new CodeAttribute(0, 0, 0, 0, code, 0, null, 0, null),
                                   instructionOffset,
                                   instructionWriter);
                //instruction.write(code, codeLength);
            }

            // Continue remapping at the next instruction offset.
            instructionOffset += instruction.length(instructionOffset);
        }

        // Correct the estimated maximum code length, now that we know the
        // actual length of this code fragment.
        maximumCodeLength += codeLength - codeFragmentOffsets[level] -
                             codeFragmentLengths[level];

        // Try to remap the exception handlers that couldn't be remapped before.
        if (allowExternalExceptionHandlers)
        {
            for (int index = 0; index < exceptionTableLength; index++)
            {
                ExceptionInfo exceptionInfo = exceptionTable[index];

                // Unmapped exception handlers are still negated.
                int handlerPC = -exceptionInfo.u2handlerPC;
                if (handlerPC > 0)
                {
                    if (remappableExceptionHandler(handlerPC))
                    {
                        exceptionInfo.u2handlerPC = newInstructionOffset(handlerPC);
                    }
                    else if (level == 0)
                    {
                        throw new IllegalStateException("Couldn't remap exception handler offset ["+handlerPC+"]");
                    }
                }
            }
        }

        level--;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        if (DEBUG)
        {
            System.out.println("CodeAttributeComposer: putting results in ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]");
        }

        if (level != -1)
        {
            throw new IllegalArgumentException("Code fragment not ended ["+level+"]");
        }

        level++;

        // Make sure the code attribute has sufficient space for the composed
        // code.
        if (codeAttribute.u4codeLength < codeLength)
        {
            codeAttribute.code = new byte[codeLength];
        }

        // Copy the composed code over into the code attribute.
        codeAttribute.u4codeLength = codeLength;
        System.arraycopy(code, 0, codeAttribute.code, 0, codeLength);

        // Remove exceptions with empty code blocks (done before).
        //exceptionTableLength =
        //    removeEmptyExceptions(exceptionTable, exceptionTableLength);

        // Make sure the exception table has sufficient space for the composed
        // exceptions.
        if (codeAttribute.exceptionTable.length < exceptionTableLength)
        {
            codeAttribute.exceptionTable = new ExceptionInfo[exceptionTableLength];
        }

        // Copy the exception table.
        codeAttribute.u2exceptionTableLength = exceptionTableLength;
        System.arraycopy(exceptionTable, 0, codeAttribute.exceptionTable, 0, exceptionTableLength);

        // Update the maximum stack size and local variable frame size.
        stackSizeUpdater.visitCodeAttribute(clazz, method, codeAttribute);
        variableSizeUpdater.visitCodeAttribute(clazz, method, codeAttribute);

        // Add a new line number table for the line numbers, if necessary.
        if (lineNumberTableLength > 0 &&
            codeAttribute.getAttribute(clazz, ClassConstants.ATTR_LineNumberTable) == null)
        {
            int attributeNameIndex =
                new ConstantPoolEditor((ProgramClass)clazz)
                    .addUtf8Constant(ClassConstants.ATTR_LineNumberTable);

            new AttributesEditor((ProgramClass)clazz, (ProgramMember)method, codeAttribute, false)
                .addAttribute(new LineNumberTableAttribute(attributeNameIndex, 0, null));
        }

        // Copy the line number table and the local variable table.
        codeAttribute.attributesAccept(clazz, method, this);

        // Remap the exception table (done before).
        //codeAttribute.exceptionsAccept(clazz, method, this);

        // Remove exceptions with empty code blocks (done before).
        //codeAttribute.u2exceptionTableLength =
        //    removeEmptyExceptions(codeAttribute.exceptionTable,
        //                          codeAttribute.u2exceptionTableLength);

        // Make sure instructions are widened if necessary.
        instructionWriter.visitCodeAttribute(clazz, method, codeAttribute);

        level--;

        if (DEBUG)
        {
            codeAttribute.accept(clazz, method, new ClassPrinter());
        }
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        // Remap all stack map entries.
        expectedStackMapFrameOffset = -1;
        stackMapAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        // Remap all stack map table entries.
        expectedStackMapFrameOffset = 0;
        stackMapTableAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        // Didn't we get line number new definitions?
        if (lineNumberTableLength == 0)
        {
            // Remap all line number table entries of the existing table.
            lineNumberTableAttribute.lineNumbersAccept(clazz, method, codeAttribute, this);
        }
        else
        {
            // Remove line numbers with empty code blocks.
            // Actually, we'll do this elsewhere, to allow processing the
            // line numbers of inlined methods.
            //lineNumberTableLength =
            //    removeEmptyLineNumbers(lineNumberTable,
            //                           lineNumberTableLength,
            //                           codeAttribute.u4codeLength);

            // Copy the line number table.
            lineNumberTableAttribute.lineNumberTable         = new LineNumberInfo[lineNumberTableLength];
            lineNumberTableAttribute.u2lineNumberTableLength = lineNumberTableLength;
            System.arraycopy(lineNumberTable, 0, lineNumberTableAttribute.lineNumberTable, 0, lineNumberTableLength);
        }
    }

    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        // Remap all local variable table entries.
        localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);

        // Remove local variables with empty code blocks.
        localVariableTableAttribute.u2localVariableTableLength =
            removeEmptyLocalVariables(localVariableTableAttribute.localVariableTable,
                                      localVariableTableAttribute.u2localVariableTableLength,
                                      codeAttribute.u2maxLocals);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        // Remap all local variable table entries.
        localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);

        // Remove local variables with empty code blocks.
        localVariableTypeTableAttribute.u2localVariableTypeTableLength =
            removeEmptyLocalVariableTypes(localVariableTypeTableAttribute.localVariableTypeTable,
                                          localVariableTypeTableAttribute.u2localVariableTypeTableLength,
                                          codeAttribute.u2maxLocals);
    }


    public void visitAnyTypeAnnotationsAttribute(Clazz clazz, TypeAnnotationsAttribute typeAnnotationsAttribute)
    {
        typeAnnotationsAttribute.typeAnnotationsAccept(clazz, this);
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
    {
        try
        {
            // Adjust the branch offset.
            branchInstruction.branchOffset =
                newBranchOffset(offset, branchInstruction.branchOffset);

            // Don't remap this instruction again.
            oldInstructionOffsets[offset] = -1;
        }
        catch (IllegalArgumentException e)
        {
            if (level == 0 || !allowExternalBranchTargets)
            {
                 throw e;
            }
        }
    }


    public void visitAnySwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SwitchInstruction switchInstruction)
    {
        try
        {
            // TODO: We're assuming we can adjust no offsets or all offsets at once.
            // Adjust the default jump offset.
            switchInstruction.defaultOffset =
                newBranchOffset(offset, switchInstruction.defaultOffset);

            // Adjust the jump offsets.
            updateJumpOffsets(offset,
                              switchInstruction.jumpOffsets);

            // Don't remap this instruction again.
            oldInstructionOffsets[offset] = -1;
        }
        catch (IllegalArgumentException e)
        {
            if (level == 0 || !allowExternalBranchTargets)
            {
                 throw e;
            }
        }
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        // Remap the code offsets. Note that the instruction offset map also has
        // an entry for the first offset after the code, for u2endPC.
        exceptionInfo.u2startPC = newInstructionOffset(exceptionInfo.u2startPC);
        exceptionInfo.u2endPC   = newInstructionOffset(exceptionInfo.u2endPC);

        // See if we can remap the handler right away. Unmapped exception
        // handlers are negated, in order to mark them as external.
        int handlerPC = exceptionInfo.u2handlerPC;
        exceptionInfo.u2handlerPC =
            !allowExternalExceptionHandlers ||
            remappableExceptionHandler(handlerPC) ?
                newInstructionOffset(handlerPC) :
                -handlerPC;
    }


    // Implementations for StackMapFrameVisitor.

    public void visitAnyStackMapFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, StackMapFrame stackMapFrame)
    {
        // Remap the stack map frame offset.
        int stackMapFrameOffset = newInstructionOffset(offset);

        int offsetDelta = stackMapFrameOffset;

        // Compute the offset delta if the frame is part of a stack map frame
        // table (for JDK 6.0) instead of a stack map (for Java Micro Edition).
        if (expectedStackMapFrameOffset >= 0)
        {
            offsetDelta -= expectedStackMapFrameOffset;

            expectedStackMapFrameOffset = stackMapFrameOffset + 1;
        }

        stackMapFrame.u2offsetDelta = offsetDelta;
    }


    public void visitSameOneFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SameOneFrame sameOneFrame)
    {
        // Remap the stack map frame offset.
        visitAnyStackMapFrame(clazz, method, codeAttribute, offset, sameOneFrame);

        // Remap the verification type offset.
        sameOneFrame.stackItemAccept(clazz, method, codeAttribute, offset, this);
    }


    public void visitMoreZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, MoreZeroFrame moreZeroFrame)
    {
        // Remap the stack map frame offset.
        visitAnyStackMapFrame(clazz, method, codeAttribute, offset, moreZeroFrame);

        // Remap the verification type offsets.
        moreZeroFrame.additionalVariablesAccept(clazz, method, codeAttribute, offset, this);
    }


    public void visitFullFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, FullFrame fullFrame)
    {
        // Remap the stack map frame offset.
        visitAnyStackMapFrame(clazz, method, codeAttribute, offset, fullFrame);

        // Remap the verification type offsets.
        fullFrame.variablesAccept(clazz, method, codeAttribute, offset, this);
        fullFrame.stackAccept(clazz, method, codeAttribute, offset, this);
    }


    // Implementations for VerificationTypeVisitor.

    public void visitAnyVerificationType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VerificationType verificationType) {}


    public void visitUninitializedType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, UninitializedType uninitializedType)
    {
        // Remap the offset of the 'new' instruction.
        uninitializedType.u2newInstructionOffset = newInstructionOffset(uninitializedType.u2newInstructionOffset);
    }


    // Implementations for LineNumberInfoVisitor.

    public void visitLineNumberInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberInfo lineNumberInfo)
    {
        // Remap the code offset.
        lineNumberInfo.u2startPC = newInstructionOffset(lineNumberInfo.u2startPC);
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        // Remap the code offset and length.
        // TODO: The local variable frame might not be strictly preserved.
        int startPC = newInstructionOffset(localVariableInfo.u2startPC);
        int endPC   = newInstructionOffset(localVariableInfo.u2startPC +
                                           localVariableInfo.u2length);

        localVariableInfo.u2startPC = startPC;
        localVariableInfo.u2length  = endPC - startPC;
    }

    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        // Remap the code offset and length.
        // TODO: The local variable frame might not be strictly preserved.
        int startPC = newInstructionOffset(localVariableTypeInfo.u2startPC);
        int endPC   = newInstructionOffset(localVariableTypeInfo.u2startPC +
                                           localVariableTypeInfo.u2length);

        localVariableTypeInfo.u2startPC = startPC;
        localVariableTypeInfo.u2length  = endPC - startPC;
    }


    // Implementations for TypeAnnotationVisitor.

    public void visitTypeAnnotation(Clazz clazz, TypeAnnotation typeAnnotation)
    {
        // Update all local variable targets.
        typeAnnotation.targetInfoAccept(clazz, this);
    }


    // Implementations for TargetInfoVisitor.

    public void visitAnyTargetInfo(Clazz clazz, TypeAnnotation typeAnnotation, TargetInfo targetInfo) {}


    public void visitLocalVariableTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, LocalVariableTargetInfo localVariableTargetInfo)
    {
        // Update the offsets of the variables.
        localVariableTargetInfo.targetElementsAccept(clazz, method, codeAttribute, typeAnnotation, this);
    }


    public void visitOffsetTargetInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, OffsetTargetInfo offsetTargetInfo)
    {
        // Update the offset.
        offsetTargetInfo.u2offset = newInstructionOffset(offsetTargetInfo.u2offset);
    }


    // Implementations for LocalVariableTargetElementVisitor.

    public void visitLocalVariableTargetElement(Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, LocalVariableTargetInfo localVariableTargetInfo, LocalVariableTargetElement localVariableTargetElement)
    {
        // Remap the code offset and length.
        int startPC = newInstructionOffset(localVariableTargetElement.u2startPC);
        int endPC   = newInstructionOffset(localVariableTargetElement.u2startPC +
                                           localVariableTargetElement.u2length);

        localVariableTargetElement.u2startPC = startPC;
        localVariableTargetElement.u2length  = endPC - startPC;
    }

    // Small utility methods.

    /**
     * Make sure the code arrays have at least the given size.
     */
    private void ensureCodeLength(int newCodeLength)
    {
        if (code.length < newCodeLength)
        {
            // Add 20% to avoid extending the arrays too often.
            newCodeLength = newCodeLength * 6 / 5;

            code                  = ArrayUtil.extendArray(code,                  newCodeLength);
            oldInstructionOffsets = ArrayUtil.extendArray(oldInstructionOffsets, newCodeLength);

            instructionWriter.extend(newCodeLength);
        }
    }


    /**
     * Adjusts the given jump offsets for the instruction at the given offset.
     */
    private void updateJumpOffsets(int offset, int[] jumpOffsets)
    {
        for (int index = 0; index < jumpOffsets.length; index++)
        {
            jumpOffsets[index] = newBranchOffset(offset, jumpOffsets[index]);
        }
    }


    /**
     * Computes the new branch offset for the instruction at the given new offset
     * with the given old branch offset.
     */
    private int newBranchOffset(int newInstructionOffset, int oldBranchOffset)
    {
        if (newInstructionOffset < 0 ||
            newInstructionOffset > codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+newInstructionOffset +"] in code with length ["+codeLength+"]");
        }

        int oldInstructionOffset = oldInstructionOffsets[newInstructionOffset];

        // For ordinary branch instructions, we can compute the offset
        // relative to the instruction itself.
        return newInstructionOffset(oldInstructionOffset + oldBranchOffset) -
               newInstructionOffset;
    }


    /**
     * Computes the new instruction offset for the instruction at the given old
     * offset.
     */
    private int newInstructionOffset(int oldInstructionOffset)
    {
        if (oldInstructionOffset < 0 ||
            oldInstructionOffset > codeFragmentLengths[level])
        {
            throw new IllegalArgumentException("Instruction offset ["+oldInstructionOffset +"] out of range in code fragment with length ["+codeFragmentLengths[level]+"] at level "+level);
        }

        int newInstructionOffset = instructionOffsetMap[level][oldInstructionOffset];
        if (newInstructionOffset == INVALID)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+oldInstructionOffset +"] in code fragment at level "+level);
        }

        return newInstructionOffset;
    }


    /**
     * Returns whether the given old exception handler can be remapped in the
     * current code fragment.
     */
    private boolean remappableExceptionHandler(int oldInstructionOffset)
    {
        // Can we index in the array?
        if (oldInstructionOffset > codeFragmentLengths[level])
        {
            return false;
        }

        // Do we have a valid new instruction offset, but not yet right after
        // the code? That offset is only labeled for mapping try blocks, not
        // for mapping handlers.
        int newInstructionOffset =
            instructionOffsetMap[level][oldInstructionOffset];

        return newInstructionOffset > INVALID &&
               newInstructionOffset < codeLength;
    }


    /**
     * Returns the given list of exceptions, without the ones that have empty
     * code blocks.
     */
    private int removeEmptyExceptions(ExceptionInfo[] exceptionInfos,
                                      int             exceptionInfoCount)
    {
        // Overwrite all empty exceptions.
        int newIndex = 0;
        for (int index = 0; index < exceptionInfoCount; index++)
        {
            ExceptionInfo exceptionInfo = exceptionInfos[index];
            if (exceptionInfo.u2startPC < exceptionInfo.u2endPC)
            {
                exceptionInfos[newIndex++] = exceptionInfo;
            }
        }

        // Clear the unused array entries.
        Arrays.fill(exceptionInfos, newIndex, exceptionInfoCount, null);

        return newIndex;
    }


    /**
     * Returns the given list of line numbers, without the ones that have empty
     * code blocks or that exceed the code size.
     */
    private int removeEmptyLineNumbers(LineNumberInfo[] lineNumberInfos,
                                       int              lineNumberInfoCount,
                                       int              codeLength)
    {
        // Overwrite all empty line number entries.
        int newIndex = 0;
        for (int index = 0; index < lineNumberInfoCount; index++)
        {
            LineNumberInfo lineNumberInfo = lineNumberInfos[index];
            int startPC = lineNumberInfo.u2startPC;
            if (startPC < codeLength &&
                (index == 0 || startPC > lineNumberInfos[index-1].u2startPC))
            {
                lineNumberInfos[newIndex++] = lineNumberInfo;
            }
        }

        // Clear the unused array entries.
        Arrays.fill(lineNumberInfos, newIndex, lineNumberInfoCount, null);

        return newIndex;
    }


    /**
     * Returns the given list of local variables, without the ones that have empty
     * code blocks or that exceed the actual number of local variables.
     */
    private int removeEmptyLocalVariables(LocalVariableInfo[] localVariableInfos,
                                          int                 localVariableInfoCount,
                                          int                 maxLocals)
    {
        // Overwrite all empty local variable entries.
        int newIndex = 0;
        for (int index = 0; index < localVariableInfoCount; index++)
        {
            LocalVariableInfo localVariableInfo = localVariableInfos[index];
            if (localVariableInfo.u2length > 0 &&
                localVariableInfo.u2index < maxLocals)
            {
                localVariableInfos[newIndex++] = localVariableInfo;
            }
        }

        // Clear the unused array entries.
        Arrays.fill(localVariableInfos, newIndex, localVariableInfoCount, null);

        return newIndex;
    }


    /**
     * Returns the given list of local variable types, without the ones that
     * have empty code blocks or that exceed the actual number of local variables.
     */
    private int removeEmptyLocalVariableTypes(LocalVariableTypeInfo[] localVariableTypeInfos,
                                              int                     localVariableTypeInfoCount,
                                              int                     maxLocals)
    {
        // Overwrite all empty local variable type entries.
        int newIndex = 0;
        for (int index = 0; index < localVariableTypeInfoCount; index++)
        {
            LocalVariableTypeInfo localVariableTypeInfo = localVariableTypeInfos[index];
            if (localVariableTypeInfo.u2length > 0 &&
                localVariableTypeInfo.u2index < maxLocals)
            {
                localVariableTypeInfos[newIndex++] = localVariableTypeInfo;
            }
        }

        // Clear the unused array entries.
        Arrays.fill(localVariableTypeInfos, newIndex, localVariableTypeInfoCount, null);

        return newIndex;
    }


    private void println(String string1, String string2)
    {
        print(string1, string2);

        System.out.println();
    }

    private void print(String string1, String string2)
    {
        System.out.print(string1);

        for (int index = 0; index < level; index++)
        {
            System.out.print("  ");
        }

        System.out.print(string2);
    }


    public static void main(String[] args)
    {
        CodeAttributeComposer composer = new CodeAttributeComposer();

        composer.beginCodeFragment(4);
        composer.appendInstruction(0, new SimpleInstruction(InstructionConstants.OP_ICONST_0));
        composer.appendInstruction(1, new VariableInstruction(InstructionConstants.OP_ISTORE, 0));
        composer.appendInstruction(2, new BranchInstruction(InstructionConstants.OP_GOTO, 1));

        composer.beginCodeFragment(4);
        composer.appendInstruction(0, new VariableInstruction(InstructionConstants.OP_IINC, 0, 1));
        composer.appendInstruction(1, new VariableInstruction(InstructionConstants.OP_ILOAD, 0));
        composer.appendInstruction(2, new SimpleInstruction(InstructionConstants.OP_ICONST_5));
        composer.appendInstruction(3, new BranchInstruction(InstructionConstants.OP_IFICMPLT, -3));
        composer.endCodeFragment();

        composer.appendInstruction(3, new SimpleInstruction(InstructionConstants.OP_RETURN));
        composer.endCodeFragment();
    }
}
