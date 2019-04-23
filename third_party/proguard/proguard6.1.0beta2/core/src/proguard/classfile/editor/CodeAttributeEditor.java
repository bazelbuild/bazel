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
import proguard.util.ArrayUtil;

import java.util.*;

/**
 * This AttributeVisitor accumulates specified changes to code, and then applies
 * these accumulated changes to the code attributes that it visits.
 *
 * The class also supports labels ({@link #label()}) and exception handlers
 * ({@link #catch_(int,int,int)}) in replacement sequences. They provide
 * local branch offsets inside the replacement sequences
 * ({@link Label#offset()}). For example, creating a replacement sequence
 * with the help of {@link InstructionSequenceBuilder}:
 * <code>
 *     final CodeAttributeEditor.Label TRY_START = codeAttributeEditor.label();
 *     final CodeAttributeEditor.Label TRY_END   = codeAttributeEditor.label();
 *     final CodeAttributeEditor.Label CATCH_END = codeAttributeEditor.label();
 *
 *     final CodeAttributeEditor.Label CATCH_EXCEPTION =
 *         codeAttributeEditor.catch_(TRY_START.offset(),
 *                                    TRY_END.offset(),
 *                                    constantPoolEditor.addClassConstant("java/lang/Exception", null));
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
 * @author Eric Lafortune
 */
public class CodeAttributeEditor
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
    public  static       boolean DEBUG = System.getProperty("cae") != null;
    //*/

    private static final int LABEL_FLAG = 0x20000000;


    private final boolean updateFrameSizes;
    private final boolean shrinkInstructions;

    private int     codeLength;
    private boolean modified;
    private boolean simple;

    private Map labels = new HashMap();

    /*private*/public Instruction[]    preOffsetInsertions = new Instruction[ClassConstants.TYPICAL_CODE_LENGTH];
    /*private*/public Instruction[]    preInsertions       = new Instruction[ClassConstants.TYPICAL_CODE_LENGTH];
    /*private*/public Instruction[]    replacements        = new Instruction[ClassConstants.TYPICAL_CODE_LENGTH];
    /*private*/public Instruction[]    postInsertions      = new Instruction[ClassConstants.TYPICAL_CODE_LENGTH];
    /*private*/public boolean[]        deleted             = new boolean[ClassConstants.TYPICAL_CODE_LENGTH];


    private int[]   newInstructionOffsets = new int[ClassConstants.TYPICAL_CODE_LENGTH];
    private int     newOffset;
    private boolean lengthIncreased;

    private int expectedStackMapFrameOffset;

    private final StackSizeUpdater    stackSizeUpdater    = new StackSizeUpdater();
    private final VariableSizeUpdater variableSizeUpdater = new VariableSizeUpdater();
    private final InstructionWriter   instructionWriter   = new InstructionWriter();


    /**
     * Creates a new CodeAttributeEditor that automatically updates frame
     * sizes and shrinks instructions.
     */
    public CodeAttributeEditor()
    {
        this(true, true);
    }


    /**
     * Creates a new CodeAttributeEditor.
     * @param updateFrameSizes   specifies whether frame sizes of edited code
     *                           should be updated.
     * @param shrinkInstructions specifies whether added instructions should
     *                           automatically be shrunk before being written.
     */
    public CodeAttributeEditor(boolean updateFrameSizes,
                               boolean shrinkInstructions)
    {
        this.updateFrameSizes   = updateFrameSizes;
        this.shrinkInstructions = shrinkInstructions;
    }


    /**
     * Resets the accumulated code changes for a given anticipated maximum
     * code length. If necessary, the size may still be extended while
     * editing the code, with {@link #extend(int)}.
     * @param codeLength the length of the code that will be edited next.
     */
    public void reset(int codeLength)
    {
        labels.clear();

        // Try to reuse the previous arrays.
        if (preInsertions.length < codeLength)
        {
            preOffsetInsertions = new Instruction[codeLength];
            preInsertions       = new Instruction[codeLength];
            replacements        = new Instruction[codeLength];
            postInsertions      = new Instruction[codeLength];
            deleted             = new boolean[codeLength];
        }
        else
        {
            Arrays.fill(preOffsetInsertions, 0, codeLength, null);
            Arrays.fill(preInsertions,       0, codeLength, null);
            Arrays.fill(replacements,        0, codeLength, null);
            Arrays.fill(postInsertions,      0, codeLength, null);
            Arrays.fill(deleted,             0, codeLength, false);
        }

        this.codeLength = codeLength;

        modified = false;
        simple   = true;
    }


    /**
     * Extends the size of the accumulated code changes.
     * @param codeLength the length of the code that will be edited next.
     */
    public void extend(int codeLength)
    {
        // Try to reuse the previous arrays.
        if (preInsertions.length < codeLength)
        {
            preOffsetInsertions = ArrayUtil.extendArray(preOffsetInsertions, codeLength);
            preInsertions       = ArrayUtil.extendArray(preInsertions,       codeLength);
            replacements        = ArrayUtil.extendArray(replacements,        codeLength);
            postInsertions      = ArrayUtil.extendArray(postInsertions,      codeLength);
            deleted             = ArrayUtil.extendArray(deleted,             codeLength);
        }
        else
        {
            Arrays.fill(preOffsetInsertions, this.codeLength, codeLength, null);
            Arrays.fill(preInsertions,       this.codeLength, codeLength, null);
            Arrays.fill(replacements,        this.codeLength, codeLength, null);
            Arrays.fill(postInsertions,      this.codeLength, codeLength, null);
            Arrays.fill(deleted,             this.codeLength, codeLength, false);
        }

        this.codeLength = codeLength;
    }


    /**
     * Remembers to place the given instruction right before the instruction
     * at the given offset. Any branches to the existing instruction will
     * land after the new instruction. Similarly, any try blocks that start at
     * the existing instruction will not include the new instruction. However,
     * any try blocks that end right before the existing instruction wil now
     * include the new instruction.
     * @param instructionOffset the offset of the instruction.
     * @param instruction       the new instruction.
     */
    public void insertBeforeOffset(int instructionOffset, Instruction instruction)
    {
        if (instructionOffset < 0 ||
            instructionOffset >= codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+instructionOffset+"] in code with length ["+codeLength+"]");
        }

        preOffsetInsertions[instructionOffset] = shrinkInstructions ?
            instruction.shrink() :
            instruction;

        modified = true;
        simple   = false;
    }


    /**
     * Remembers to place the given instruction right before the instruction
     * at the given offset. Any branches to the existing instruction will
     * also go to the new instruction. Similarly, any try blocks that include
     * the existing instruction will also include the new instruction.
     * @param instructionOffset the offset of the instruction.
     * @param instruction       the new instruction.
     */
    public void insertBeforeInstruction(int instructionOffset, Instruction instruction)
    {
        if (instructionOffset < 0 ||
            instructionOffset >= codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+instructionOffset+"] in code with length ["+codeLength+"]");
        }

        preInsertions[instructionOffset] = shrinkInstructions ?
            instruction.shrink() :
            instruction;

        modified = true;
        simple   = false;
    }


    /**
     * Remembers to place the given instructions right before the instruction
     * at the given offset. Any branches to the existing instruction will
     * land after the new instructions. Similarly, any try blocks that start at
     * the existing instruction will not include the new instructions. However,
     * any try blocks that end right before the existing instruction wil now
     * include the new instructions.
     * @param instructionOffset the offset of the instruction.
     * @param instructions      the new instructions.
     */
    public void insertBeforeOffset(int instructionOffset, Instruction[] instructions)
    {
        if (instructionOffset < 0 ||
            instructionOffset >= codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+instructionOffset+"] in code with length ["+codeLength+"]");
        }

        CompositeInstruction instruction =
            new CompositeInstruction(instructions);

        preOffsetInsertions[instructionOffset] = shrinkInstructions ?
            instruction.shrink() :
            instruction;

        modified = true;
        simple   = false;
    }


    /**
     * Remembers to place the given instructions right before the instruction
     * at the given offset. Any branches to the existing instruction will
     * also go to the new instructions. Similarly, any try blocks that include
     * the existing instruction will also include the new instructions.
     * @param instructionOffset the offset of the instruction.
     * @param instructions      the new instructions.
     */
    public void insertBeforeInstruction(int instructionOffset, Instruction[] instructions)
    {
        if (instructionOffset < 0 ||
            instructionOffset >= codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+instructionOffset+"] in code with length ["+codeLength+"]");
        }

        CompositeInstruction instruction =
            new CompositeInstruction(instructions);

        preInsertions[instructionOffset] = shrinkInstructions ?
            instruction.shrink() :
            instruction;

        modified = true;
        simple   = false;
    }


    /**
     * Remembers to replace the instruction at the given offset by the given
     * instruction.
     * @param instructionOffset the offset of the instruction to be replaced.
     * @param instruction       the new instruction.
     */
    public void replaceInstruction(int instructionOffset, Instruction instruction)
    {
        if (instructionOffset < 0 ||
            instructionOffset >= codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+instructionOffset+"] in code with length ["+codeLength+"]");
        }

        replacements[instructionOffset] = shrinkInstructions ?
            instruction.shrink() :
            instruction;

        modified = true;
    }


    /**
     * Remembers to replace the instruction at the given offset by the given
     * instructions.
     * @param instructionOffset the offset of the instruction to be replaced.
     * @param instructions      the new instructions.
     */
    public void replaceInstruction(int instructionOffset, Instruction[] instructions)
    {
        if (instructionOffset < 0 ||
            instructionOffset >= codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+instructionOffset+"] in code with length ["+codeLength+"]");
        }

        CompositeInstruction instruction =
            new CompositeInstruction(instructions);

        replacements[instructionOffset] = shrinkInstructions ?
            instruction.shrink() :
            instruction;

        modified = true;
    }


    /**
     * Remembers to place the given instruction right after the instruction
     * at the given offset.
     * @param instructionOffset the offset of the instruction.
     * @param instruction       the new instruction.
     */
    public void insertAfterInstruction(int instructionOffset, Instruction instruction)
    {
        if (instructionOffset < 0 ||
            instructionOffset >= codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+instructionOffset+"] in code with length ["+codeLength+"]");
        }

        postInsertions[instructionOffset] = shrinkInstructions ?
            instruction.shrink() :
            instruction;

        modified = true;
        simple   = false;
    }


    /**
     * Remembers to place the given instructions right after the instruction
     * at the given offset.
     * @param instructionOffset the offset of the instruction.
     * @param instructions      the new instructions.
     */
    public void insertAfterInstruction(int instructionOffset, Instruction[] instructions)
    {
        if (instructionOffset < 0 ||
            instructionOffset >= codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+instructionOffset+"] in code with length ["+codeLength+"]");
        }

        CompositeInstruction instruction =
            new CompositeInstruction(instructions);

        postInsertions[instructionOffset] = shrinkInstructions ?
            instruction.shrink() :
            instruction;

        modified = true;
        simple   = false;
    }


    /**
     * Remembers to delete the instruction at the given offset.
     * @param instructionOffset the offset of the instruction to be deleted.
     */
    public void deleteInstruction(int instructionOffset)
    {
        if (instructionOffset < 0 ||
            instructionOffset >= codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+instructionOffset+"] in code with length ["+codeLength+"]");
        }

        deleted[instructionOffset] = true;

        modified = true;
        simple   = false;
    }


    /**
     * Remembers not to delete the instruction at the given offset.
     * @param instructionOffset the offset of the instruction not to be deleted.
     */
    public void undeleteInstruction(int instructionOffset)
    {
        if (instructionOffset < 0 ||
            instructionOffset >= codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+instructionOffset+"] in code with length ["+codeLength+"]");
        }

        deleted[instructionOffset] = false;
    }


    /**
     * Clears all modifications of the instruction at the given offset.
     * @param instructionOffset the offset of the instruction to be deleted.
     */
    public void clearModifications(int instructionOffset)
    {
        if (instructionOffset < 0 ||
            instructionOffset >= codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+instructionOffset+"] in code with length ["+codeLength+"]");
        }

        preOffsetInsertions[instructionOffset] = null;
        preInsertions[instructionOffset]       = null;
        replacements[instructionOffset]        = null;
        postInsertions[instructionOffset]      = null;
        deleted[instructionOffset]             = false;
    }


    /**
     * Returns whether the code has been modified in any way.
     */
    public boolean isModified()
    {
        return modified;
    }


    /**
     * Returns whether the instruction at the given offset has been modified
     * in any way.
     */
    public boolean isModified(int instructionOffset)
    {
        return preOffsetInsertions[instructionOffset] != null ||
               preInsertions[instructionOffset]       != null ||
               replacements[instructionOffset]        != null ||
               postInsertions[instructionOffset]      != null ||
               deleted[instructionOffset];
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
//        DEBUG =
//            clazz.getName().equals("abc/Def") &&
//            method.getName(clazz).equals("abc");

        // TODO: Remove this when the code has stabilized.
        // Catch any unexpected exceptions from the actual visiting method.
        try
        {
            // Process the code.
            visitCodeAttribute0(clazz, method, codeAttribute);
        }
        catch (RuntimeException ex)
        {
            System.err.println("Unexpected error while editing code:");
            System.err.println("  Class       = ["+clazz.getName()+"]");
            System.err.println("  Method      = ["+method.getName(clazz)+method.getDescriptor(clazz)+"]");
            System.err.println("  Exception   = ["+ex.getClass().getName()+"] ("+ex.getMessage()+")");

            throw ex;
        }
    }


    public void visitCodeAttribute0(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Do we have to update the code?
        if (modified)
        {
            if (DEBUG)
            {
                System.out.println("CodeAttributeEditor: "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz));
            }

            // Can we perform a faster simple replacement of instructions?
            if (canPerformSimpleReplacements(codeAttribute))
            {
                if (DEBUG)
                {
                    System.out.println("  Simple editing");
                }

                // Simply overwrite the instructions.
                performSimpleReplacements(codeAttribute);
            }
            else
            {
                if (DEBUG)
                {
                    System.out.println("  Full editing");
                }

                // Move and remap the instructions.
                codeAttribute.u4codeLength =
                    updateInstructions(clazz, method, codeAttribute);

                // Update the exception table.
                codeAttribute.exceptionsAccept(clazz, method, this);

                // Remove exceptions with empty code blocks.
                codeAttribute.u2exceptionTableLength =
                    removeEmptyExceptions(codeAttribute.exceptionTable,
                                          codeAttribute.u2exceptionTableLength);

                // Update the line number table and the local variable tables.
                codeAttribute.attributesAccept(clazz, method, this);
            }

            // Make sure instructions are widened if necessary.
            instructionWriter.visitCodeAttribute(clazz, method, codeAttribute);
        }

        // Update the maximum stack size and local variable frame size.
        if (updateFrameSizes)
        {
            stackSizeUpdater.visitCodeAttribute(clazz, method, codeAttribute);
            variableSizeUpdater.visitCodeAttribute(clazz, method, codeAttribute);
        }
    }


    public void visitStackMapAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapAttribute stackMapAttribute)
    {
        // Update all stack map entries.
        expectedStackMapFrameOffset = -1;
        stackMapAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
    }


    public void visitStackMapTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, StackMapTableAttribute stackMapTableAttribute)
    {
        // Update all stack map table entries.
        expectedStackMapFrameOffset = 0;
        stackMapTableAttribute.stackMapFramesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        // Update all line number table entries.
        lineNumberTableAttribute.lineNumbersAccept(clazz, method, codeAttribute, this);

        // Remove line numbers with empty code blocks.
//        lineNumberTableAttribute.u2lineNumberTableLength =
//           removeEmptyLineNumbers(lineNumberTableAttribute.lineNumberTable,
//                                  lineNumberTableAttribute.u2lineNumberTableLength,
//                                  codeAttribute.u4codeLength);
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        // Update all local variable table entries.
        localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        // Update all local variable table entries.
        localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitAnyTypeAnnotationsAttribute(Clazz clazz, TypeAnnotationsAttribute typeAnnotationsAttribute)
    {
        typeAnnotationsAttribute.typeAnnotationsAccept(clazz, this);
    }


    /**
     * Checks if it is possible to modifies the given code without having to
     * update any offsets.
     * @param codeAttribute the code to be changed.
     * @return the new code length.
     */
    private boolean canPerformSimpleReplacements(CodeAttribute codeAttribute)
    {
        if (!simple)
        {
            return false;
        }

        byte[] code       = codeAttribute.code;
        int    codeLength = codeAttribute.u4codeLength;

        // Go over all replacement instructions.
        for (int offset = 0; offset < codeLength; offset++)
        {
            // Check if the replacement instruction, if any, has a different
            // length than the original instruction.
            Instruction replacementInstruction = replacements[offset];
            if (replacementInstruction != null &&
                replacementInstruction.length(offset) !=
                    InstructionFactory.create(code, offset).length(offset))
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Modifies the given code without updating any offsets.
     * @param codeAttribute the code to be changed.
     */
    private void performSimpleReplacements(CodeAttribute codeAttribute)
    {
        int codeLength = codeAttribute.u4codeLength;

        // Go over all replacement instructions.
        for (int offset = 0; offset < codeLength; offset++)
        {
            // Overwrite the original instruction with the replacement
            // instruction if any.
            Instruction replacementInstruction = replacements[offset];
            if (replacementInstruction != null)
            {
                replacementInstruction.write(codeAttribute, offset);

                if (DEBUG)
                {
                    System.out.println("  Replaced "+replacementInstruction.toString(offset));
                }
            }
        }
    }


    /**
     * Modifies the given code based on the previously specified changes.
     * @param clazz         the class file of the code to be changed.
     * @param method        the method of the code to be changed.
     * @param codeAttribute the code to be changed.
     * @return the new code length.
     */
    private int updateInstructions(Clazz         clazz,
                                   Method        method,
                                   CodeAttribute codeAttribute)
    {
        byte[] oldCode   = codeAttribute.code;
        int    oldLength = codeAttribute.u4codeLength;

        // Make sure there is a sufficiently large instruction offset map.
        if (newInstructionOffsets.length < oldLength + 1)
        {
            newInstructionOffsets = new int[oldLength + 1];
        }

        // Fill out the instruction offset map.
        int newLength = mapInstructions(oldCode,
                                        oldLength);

        // Create a new code array if necessary.
        if (lengthIncreased)
        {
            codeAttribute.code = new byte[newLength];
        }

        // Prepare for possible widening of instructions.
        instructionWriter.reset(newLength);

        // Move the instructions into the new code array.
        moveInstructions(clazz,
                         method,
                         codeAttribute,
                         oldCode,
                         oldLength);

        // We can return the new length.
        return newLength;
    }


    /**
     * Fills out the instruction offset map for the given code block.
     * @param oldCode   the instructions to be moved.
     * @param oldLength the code length.
     * @return the new code length.
     */
    private int mapInstructions(byte[] oldCode, int oldLength)
    {
        // Start mapping instructions at the beginning.
        newOffset       = 0;
        lengthIncreased = false;

        int oldOffset = 0;
        do
        {
            // Get the next instruction.
            Instruction instruction = InstructionFactory.create(oldCode, oldOffset);

            // Compute the mapping of the instruction.
            mapInstruction(oldOffset, instruction);

            oldOffset += instruction.length(oldOffset);

            if (newOffset > oldOffset)
            {
                lengthIncreased = true;
            }
        }
        while (oldOffset < oldLength);

        // Also add an entry for the first offset after the code.
        newInstructionOffsets[oldOffset] = newOffset;

        return newOffset;
    }


    /**
     * Fills out the instruction offset map for the given instruction.
     * @param oldOffset   the instruction's old offset.
     * @param instruction the instruction to be moved.
     */
    private void mapInstruction(int         oldOffset,
                                Instruction instruction)
    {
        // Account for the pre-offset-inserted instruction, if any.
        Instruction preOffsetInstruction = preOffsetInsertions[oldOffset];
        if (preOffsetInstruction != null)
        {
            newOffset += preOffsetInstruction.length(newOffset);
        }

        newInstructionOffsets[oldOffset] = newOffset;

        // Account for the pre-inserted instruction, if any.
        Instruction preInstruction = preInsertions[oldOffset];
        if (preInstruction != null)
        {
            newOffset += preInstruction.length(newOffset);
        }

        // Account for the replacement instruction, or for the current
        // instruction, if it shouldn't be  deleted.
        Instruction replacementInstruction = replacements[oldOffset];
        if (replacementInstruction != null)
        {
            newOffset += replacementInstruction.length(newOffset);
        }
        else if (!deleted[oldOffset])
        {
            // Note that the instruction's length may change at its new offset,
            // e.g. if it is a switch instruction.
            newOffset += instruction.length(newOffset);
        }

        // Account for the post-inserted instruction, if any.
        Instruction postInstruction = postInsertions[oldOffset];
        if (postInstruction != null)
        {
            newOffset += postInstruction.length(newOffset);
        }
    }


    /**
     * Moves the given code block to the new offsets.
     * @param clazz         the class file of the code to be changed.
     * @param method        the method of the code to be changed.
     * @param codeAttribute the code to be changed.
     * @param oldCode       the original code to be moved.
     * @param oldLength     the original code length.
     */
    private void moveInstructions(Clazz         clazz,
                                  Method        method,
                                  CodeAttribute codeAttribute,
                                  byte[]        oldCode,
                                  int           oldLength)
    {
        // Start writing instructions at the beginning.
        newOffset = 0;

        int oldOffset = 0;
        do
        {
            // Get the next instruction.
            Instruction instruction = InstructionFactory.create(oldCode, oldOffset);

            // Move the instruction to its new offset.
            moveInstruction(clazz,
                            method,
                            codeAttribute,
                            oldOffset,
                            instruction);

            oldOffset += instruction.length(oldOffset);
        }
        while (oldOffset < oldLength);
    }


    /**
     * Moves the given instruction to its new offset.
     * @param clazz         the class file of the code to be changed.
     * @param method        the method of the code to be changed.
     * @param codeAttribute the code to be changed.
     * @param oldOffset     the original instruction offset.
     * @param instruction   the original instruction.
     */
    private void moveInstruction(Clazz         clazz,
                                 Method        method,
                                 CodeAttribute codeAttribute,
                                 int           oldOffset,
                                 Instruction   instruction)
    {
        // Update and insert the pre-inserted instruction, if any.
        Instruction preOffsetInstruction = preOffsetInsertions[oldOffset];
        if (preOffsetInstruction != null)
        {
            if (DEBUG)
            {
                System.out.println("  Pre-inserted  ["+oldOffset+"] -> "+preOffsetInstruction.toString(newOffset));
            }

            // Update the instruction.
            preOffsetInstruction.accept(clazz, method, codeAttribute, oldOffset, this);
        }

        // Update and insert the pre-inserted instruction, if any.
        Instruction preInstruction = preInsertions[oldOffset];
        if (preInstruction != null)
        {
            if (DEBUG)
            {
                System.out.println("  Pre-inserted  ["+oldOffset+"] -> "+preInstruction.toString(newOffset));
            }

            // Update the instruction.
            preInstruction.accept(clazz, method, codeAttribute, oldOffset, this);
        }

        // Update and insert the replacement instruction, or the current
        // instruction, if it shouldn't be deleted.
        Instruction replacementInstruction = replacements[oldOffset];
        if (replacementInstruction != null)
        {
            if (DEBUG)
            {
                System.out.println("  Replaced      ["+oldOffset+"] -> "+replacementInstruction.toString(newOffset));
            }

            // Update the instruction.
            replacementInstruction.accept(clazz, method, codeAttribute, oldOffset, this);
        }
        else if (!deleted[oldOffset])
        {
            if (DEBUG)
            {
                System.out.println("  Copied        ["+oldOffset+"] -> "+instruction.toString(newOffset));
            }

            // Update the instruction.
            instruction.accept(clazz, method, codeAttribute, oldOffset, this);
        }

        // Update and insert the post-inserted instruction, if any.
        Instruction postInstruction = postInsertions[oldOffset];
        if (postInstruction != null)
        {
            if (DEBUG)
            {
                System.out.println("  Post-inserted ["+oldOffset+"] -> "+postInstruction.toString(newOffset));
            }

            // Update the instruction.
            postInstruction.accept(clazz, method, codeAttribute, oldOffset, this);
        }
    }


    // Implementations for InstructionVisitor.

    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        // Write out the instruction.
        instructionWriter.visitSimpleInstruction(clazz,
                                                 method,
                                                 codeAttribute,
                                                 newOffset,
                                                 simpleInstruction);

        newOffset += simpleInstruction.length(newOffset);
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        // Write out the instruction.
        instructionWriter.visitConstantInstruction(clazz,
                                                   method,
                                                   codeAttribute,
                                                   newOffset,
                                                   constantInstruction);

        newOffset += constantInstruction.length(newOffset);
    }


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        // Write out the instruction.
        instructionWriter.visitVariableInstruction(clazz,
                                                   method,
                                                   codeAttribute,
                                                   newOffset,
                                                   variableInstruction);

        newOffset += variableInstruction.length(newOffset);
    }


    public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
    {
        // Update the branch offset, relative to the precise new offset.
        branchInstruction.branchOffset =
            newBranchOffset(offset, branchInstruction.branchOffset, newOffset);

        // Write out the instruction.
        instructionWriter.visitBranchInstruction(clazz,
                                                 method,
                                                 codeAttribute,
                                                 newOffset,
                                                 branchInstruction);

        newOffset += branchInstruction.length(newOffset);
    }


    public void visitTableSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, TableSwitchInstruction tableSwitchInstruction)
    {
        // Update the default jump offset, relative to the precise new offset.
        tableSwitchInstruction.defaultOffset =
            newBranchOffset(offset, tableSwitchInstruction.defaultOffset, newOffset);

        // Update the jump offsets, relative to the precise new offset.
        newJumpOffsets(offset,
                       tableSwitchInstruction.jumpOffsets,
                       newOffset);

        // Write out the instruction.
        instructionWriter.visitTableSwitchInstruction(clazz,
                                                      method,
                                                      codeAttribute,
                                                      newOffset,
                                                      tableSwitchInstruction);

        newOffset += tableSwitchInstruction.length(newOffset);
    }


    public void visitLookUpSwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, LookUpSwitchInstruction lookUpSwitchInstruction)
    {
        // Update the default jump offset, relative to the precise new offset.
        lookUpSwitchInstruction.defaultOffset =
            newBranchOffset(offset, lookUpSwitchInstruction.defaultOffset, newOffset);

        // Update the jump offsets, relative to the precise new offset.
        newJumpOffsets(offset,
                       lookUpSwitchInstruction.jumpOffsets,
                       newOffset);

        // Write out the instruction.
        instructionWriter.visitLookUpSwitchInstruction(clazz,
                                                       method,
                                                       codeAttribute,
                                                       newOffset,
                                                       lookUpSwitchInstruction);

        newOffset += lookUpSwitchInstruction.length(newOffset);
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        // Update the code offsets. Note that the instruction offset map also
        // has an entry for the first offset after the code, for u2endPC.
        exceptionInfo.u2startPC   = newInstructionOffset(exceptionInfo.u2startPC);
        exceptionInfo.u2endPC     = newInstructionOffset(exceptionInfo.u2endPC);
        exceptionInfo.u2handlerPC = newInstructionOffset(exceptionInfo.u2handlerPC);
    }


    // Implementations for StackMapFrameVisitor.

    public void visitAnyStackMapFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, StackMapFrame stackMapFrame)
    {
        // Update the stack map frame offset.
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
        // Update the stack map frame offset.
        visitAnyStackMapFrame(clazz, method, codeAttribute, offset, sameOneFrame);

        // Update the verification type offset.
        sameOneFrame.stackItemAccept(clazz, method, codeAttribute, offset, this);
    }


    public void visitMoreZeroFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, MoreZeroFrame moreZeroFrame)
    {
        // Update the stack map frame offset.
        visitAnyStackMapFrame(clazz, method, codeAttribute, offset, moreZeroFrame);

        // Update the verification type offsets.
        moreZeroFrame.additionalVariablesAccept(clazz, method, codeAttribute, offset, this);
    }


    public void visitFullFrame(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, FullFrame fullFrame)
    {
        // Update the stack map frame offset.
        visitAnyStackMapFrame(clazz, method, codeAttribute, offset, fullFrame);

        // Update the verification type offsets.
        fullFrame.variablesAccept(clazz, method, codeAttribute, offset, this);
        fullFrame.stackAccept(clazz, method, codeAttribute, offset, this);
    }


    // Implementations for VerificationTypeVisitor.

    public void visitAnyVerificationType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VerificationType verificationType) {}


    public void visitUninitializedType(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, UninitializedType uninitializedType)
    {
        // Update the offset of the 'new' instruction.
        uninitializedType.u2newInstructionOffset = newInstructionOffset(uninitializedType.u2newInstructionOffset);
    }


    // Implementations for LineNumberInfoVisitor.

    public void visitLineNumberInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberInfo lineNumberInfo)
    {
        // Update the code offset.
        lineNumberInfo.u2startPC = newInstructionOffset(lineNumberInfo.u2startPC);
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        // Update the code offset and length.
        // Be careful to update the length first.
        localVariableInfo.u2length  = newBranchOffset(localVariableInfo.u2startPC, localVariableInfo.u2length);
        localVariableInfo.u2startPC = newInstructionOffset(localVariableInfo.u2startPC);
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        // Update the code offset and length.
        // Be careful to update the length first.
        localVariableTypeInfo.u2length  = newBranchOffset(localVariableTypeInfo.u2startPC, localVariableTypeInfo.u2length);
        localVariableTypeInfo.u2startPC = newInstructionOffset(localVariableTypeInfo.u2startPC);
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
        // Update the variable start offset and length.
        // Be careful to update the length first.
        localVariableTargetElement.u2length  = newBranchOffset(localVariableTargetElement.u2startPC, localVariableTargetElement.u2length);
        localVariableTargetElement.u2startPC = newInstructionOffset(localVariableTargetElement.u2startPC);
    }


    // Small utility methods.

    /**
     * Updates the given jump offsets for the instruction at the given offset,
     * relative to the given new offset.
     */
    private void newJumpOffsets(int   oldInstructionOffset,
                                int[] oldJumpOffsets,
                                int   newInstructionOffset)
    {
        for (int index = 0; index < oldJumpOffsets.length; index++)
        {
            oldJumpOffsets[index] = newBranchOffset(oldInstructionOffset,
                                                    oldJumpOffsets[index],
                                                    newInstructionOffset);
        }
    }


    /**
     * Computes the new branch offset for the instruction at the given offset
     * with the given branch offset, relative to the new instruction (block)
     * offset.
     */
    private int newBranchOffset(int oldInstructionOffset,
                                int oldBranchOffset)
    {
        return newInstructionOffset(oldInstructionOffset + oldBranchOffset) -
               newInstructionOffset(oldInstructionOffset);
    }


    /**
     * Computes the new branch offset for the instruction at the given offset
     * with the given branch offset, relative to the given new offset.
     */
    private int newBranchOffset(int oldInstructionOffset,
                                int oldBranchOffset,
                                int newInstructionOffset)
    {
        // Compute the old branch target.
        // Pass a label offset unchanged.
        int oldBranchTargetOffset = isLabel(oldBranchOffset) ? oldBranchOffset :
            oldInstructionOffset + oldBranchOffset;

        return newInstructionOffset(oldBranchTargetOffset) -
               newInstructionOffset;
    }


    /**
     * Computes the new instruction offset for the instruction at the given
     * offset.
     */
    private int newInstructionOffset(int oldInstructionOffset)
    {
        // Special case: is it actually a label?
        if (isLabel(oldInstructionOffset))
        {
            // Retrieve the new offset from the label.
            int labelIdentifier = labelIdentifier(oldInstructionOffset);
            Label label = (Label)labels.get(labelIdentifier);
            if (label == null)
            {
                throw new IllegalArgumentException("Reference to unknown label identifier ["+labelIdentifier+"]");
            }

            return label.newOffset;
        }

        // Otherwise retrieve the new instruction offset.
        if (oldInstructionOffset < 0 ||
            oldInstructionOffset > codeLength)
        {
            throw new IllegalArgumentException("Invalid instruction offset ["+oldInstructionOffset+"] in code with length ["+codeLength+"]");
        }

        return newInstructionOffsets[oldInstructionOffset];
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

        return newIndex;
    }


    /**
     * This pseudo-instruction is a composite of other instructions, for local
     * use inside the editor class only.
     */
    private class CompositeInstruction
    extends       Instruction
    {
        private Instruction[] instructions;


        private CompositeInstruction(Instruction[] instructions)
        {
            this.instructions = instructions;
        }


        // Implementations for Instruction.

        public Instruction shrink()
        {
            for (int index = 0; index < instructions.length; index++)
            {
                instructions[index] = instructions[index].shrink();
            }

            return this;
        }


        public void write(byte[] code, int offset)
        {
            for (int index = 0; index < instructions.length; index++)
            {
                Instruction instruction = instructions[index];

                instruction.write(code, offset);

                offset += instruction.length(offset);
            }
        }


        protected void readInfo(byte[] code, int offset)
        {
            throw new UnsupportedOperationException("Can't read composite instruction");
        }


        protected void writeInfo(byte[] code, int offset)
        {
            throw new UnsupportedOperationException("Can't write composite instruction");
        }


        public int length(int offset)
        {
            int newOffset = offset;

            for (int index = 0; index < instructions.length; index++)
            {
                newOffset += instructions[index].length(newOffset);
            }

            return newOffset - offset;
        }


        public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, InstructionVisitor instructionVisitor)
        {
            if (instructionVisitor != CodeAttributeEditor.this)
            {
                throw new UnsupportedOperationException("Unexpected visitor ["+instructionVisitor+"]");
            }

            for (int index = 0; index < instructions.length; index++)
            {
                Instruction instruction = instructions[index];

                instruction.accept(clazz, method, codeAttribute, offset, CodeAttributeEditor.this);
            }
        }


        // Implementations for Object.

        public String toString()
        {
            StringBuffer stringBuffer = new StringBuffer();

            for (int index = 0; index < instructions.length; index++)
            {
                stringBuffer.append(instructions[index].toString()).append("; ");
            }

            return stringBuffer.toString();
        }
    }


    // For convenience, we also define two pseudo-instructions, to conveniently
    // mark local labels and create new exceptions handlers.

    /**
     * Creates a new label that can be used as a pseudo-instruction to mark
     * a local offset. Its offset can be used as a branch target in
     * replacement instructions ({@link Label#offset()}).
     */
    public Label label()
    {
        return label(labels.size());
    }


    /**
     * Creates a new label that can be used as a pseudo-instruction to mark
     * a local offset. Its offset can be used as a branch target in
     * replacement instructions ({@link Label#offset()}).
     */
    public Label label(int identifier)
    {
        Label label = new Label(identifier);

        // Remember the label, so we can retrieve its offset later on.
        labels.put(new Integer(identifier), label);

        return label;
    }


    /**
     * Creates a new catch instance that can be used as a pseudo-instruction
     * to mark the start of an exception handler. Its offset can be used as
     * a branch target in replacement instructions ({@link Label#offset()}).
     */
    public Label catch_(int startOffset,
                        int endOffset,
                        int catchType)
    {
        return catch_(labels.size(),
                      startOffset,
                      endOffset,
                      catchType);
    }


    /**
     * Creates a new catch instance that can be used as a pseudo-instruction
     * to mark the start of an exception handler. Its offset can be used as
     * a branch target in replacement instructions ({@link Label#offset()}).
     */
    public Label catch_(int identifier,
                        int startOffset,
                        int endOffset,
                        int catchType)
    {
        Label catch_ = new Catch(identifier, startOffset, endOffset, catchType);

        // Remember the label, so we can retrieve its offset later on.
        labels.put(new Integer(identifier), catch_);

        return catch_;
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
     * Returns the label identifier that corrresponds to the given
     * instruction offset.
     */
    private static int labelIdentifier(int instructionOffset)
    {
        return instructionOffset & ~LABEL_FLAG;
    }


    /**
     * This pseudo-instruction represents a label that marks an instruction
     * offset, for use in the context of the code attribute editor only.
     */
    public static class Label
    extends             Instruction
    {
        protected final int identifier;

        private int newOffset;


        /**
         * Creates a new Label.
         * @param identifier an identifier that can be chosen freely.
         */
        public Label(int identifier)
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
            // Remember the offset, so we can retrieve it later on.
            newOffset = offset;

            return 0;
        }


        public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, InstructionVisitor instructionVisitor)
        {
            if (instructionVisitor.getClass() != CodeAttributeEditor.class)
            {
                throw new UnsupportedOperationException("Unexpected visitor ["+instructionVisitor+"]");
            }
        }


        // Implementations for Object.

        public String toString()
        {
            return "label_"+offset();
        }
    }


    /**
     * This pseudo-instruction represents an exception handler,
     * for use in the context of the code attribute editor only.
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
        public Catch(int identifier,
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
            if (instructionVisitor.getClass() != CodeAttributeEditor.class)
            {
                throw new UnsupportedOperationException("Unexpected visitor ["+instructionVisitor+"]");
            }

            // Add the exception. Its offsets will still be updated later on,
            // like any other exception.
            new ExceptionInfoEditor(codeAttribute).prependException(
                new ExceptionInfo(startOfffset,
                                  endOffset,
                                  offset(),
                                  catchType));
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
