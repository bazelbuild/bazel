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
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This InstructionVisitor writes out the instructions that it visits,
 * collecting instructions that have to be widened. As an AttributeVisitor,
 * it then applies the collected changes. The process will be repeated
 * recursively, if necessary. The caller still has to update the frame sizes.
 *
 * @author Eric Lafortune
 */
public class InstructionWriter
extends      SimplifiedVisitor
implements   InstructionVisitor,
             AttributeVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    public  static       boolean DEBUG = false;
    //*/


    private int codeLength;

    private CodeAttributeEditor codeAttributeEditor;


    /**
     * Resets the accumulated code.
     * @param codeLength the length of the code that will be edited next.
     */
    public void reset(int codeLength)
    {
        this.codeLength = codeLength;

        if (codeAttributeEditor != null)
        {
            codeAttributeEditor.reset(codeLength);
        }
    }


    /**
     * Extends the size of the accumulated code.
     * @param codeLength the length of the code that will be edited next.
     */
    public void extend(int codeLength)
    {
        this.codeLength = codeLength;

        if (codeAttributeEditor != null)
        {
            codeAttributeEditor.extend(codeLength);
        }
    }


    // Implementations for InstructionVisitor.

    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        // Try to write out the instruction.
        // Simple instructions should always fit.
        simpleInstruction.write(codeAttribute, offset);
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        try
        {
            // Try to write out the instruction.
            constantInstruction.write(codeAttribute, offset);
        }
        catch (IllegalArgumentException exception)
        {
            // Create a new constant instruction that will fit.
            Instruction replacementInstruction =
                new ConstantInstruction(constantInstruction.opcode,
                                        constantInstruction.constantIndex,
                                        constantInstruction.constant);

            if (DEBUG)
            {
                System.out.println("  "+constantInstruction.toString(offset)+" will be widened to "+replacementInstruction.toString());
            }

            replaceInstruction(offset, replacementInstruction);

            // Write out a dummy constant instruction for now.
            constantInstruction.constantIndex = 0;
            constantInstruction.constant      = 0;
            constantInstruction.write(codeAttribute, offset);
        }
    }


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        try
        {
            // Try to write out the instruction.
            variableInstruction.write(codeAttribute, offset);
        }
        catch (IllegalArgumentException exception)
        {
            // Create a new variable instruction that will fit.
            Instruction replacementInstruction =
                new VariableInstruction(variableInstruction.opcode,
                                        variableInstruction.variableIndex,
                                        variableInstruction.constant);

            replaceInstruction(offset, replacementInstruction);

            if (DEBUG)
            {
                System.out.println("  "+variableInstruction.toString(offset)+" will be widened to "+replacementInstruction.toString());
            }

            // Write out a dummy variable instruction for now.
            variableInstruction.variableIndex = 0;
            variableInstruction.constant      = 0;
            variableInstruction.write(codeAttribute, offset);
        }
    }


    public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
    {
        try
        {
            // Try to write out the instruction.
            branchInstruction.write(codeAttribute, offset);
        }
        catch (IllegalArgumentException exception)
        {
            // Create a new unconditional branch that will fit.
            Instruction replacementInstruction =
                new BranchInstruction(InstructionConstants.OP_GOTO_W,
                                      branchInstruction.branchOffset);

            // Create a new instruction that will fit.
            switch (branchInstruction.opcode)
            {
                default:
                {
                    // Create a new branch instruction that will fit.
                    replacementInstruction =
                        new BranchInstruction(branchInstruction.opcode,
                                              branchInstruction.branchOffset);

                    break;
                }

                // Some special cases, for which a wide branch doesn't exist.
                case InstructionConstants.OP_IFEQ:
                case InstructionConstants.OP_IFNE:
                case InstructionConstants.OP_IFLT:
                case InstructionConstants.OP_IFGE:
                case InstructionConstants.OP_IFGT:
                case InstructionConstants.OP_IFLE:
                case InstructionConstants.OP_IFICMPEQ:
                case InstructionConstants.OP_IFICMPNE:
                case InstructionConstants.OP_IFICMPLT:
                case InstructionConstants.OP_IFICMPGE:
                case InstructionConstants.OP_IFICMPGT:
                case InstructionConstants.OP_IFICMPLE:
                case InstructionConstants.OP_IFACMPEQ:
                case InstructionConstants.OP_IFACMPNE:
                {
                    // Insert the complementary conditional branch.
                    Instruction complementaryConditionalBranch =
                        new BranchInstruction((byte)(((branchInstruction.opcode+1) ^ 1) - 1),
                                              (1+2) + (1+4));

                    insertBeforeInstruction(offset, complementaryConditionalBranch);

                    // Create a new unconditional branch that will fit.
                    break;
                }

                case InstructionConstants.OP_IFNULL:
                case InstructionConstants.OP_IFNONNULL:
                {
                    // Insert the complementary conditional branch.
                    Instruction complementaryConditionalBranch =
                        new BranchInstruction((byte)(branchInstruction.opcode ^ 1),
                                              (1+2) + (1+4));

                    insertBeforeInstruction(offset, complementaryConditionalBranch);

                    // Create a new unconditional branch that will fit.
                    break;
                }
            }

            if (DEBUG)
            {
                System.out.println("  "+branchInstruction.toString(offset)+" will be widened to "+replacementInstruction.toString());
            }

            replaceInstruction(offset, replacementInstruction);

            // Write out a dummy branch instruction for now.
            branchInstruction.branchOffset = 0;
            branchInstruction.write(codeAttribute, offset);
        }
    }


    public void visitAnySwitchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SwitchInstruction switchInstruction)
    {
        // Try to write out the instruction.
        // Switch instructions should always fit.
        switchInstruction.write(codeAttribute, offset);
    }


    // Implementations for AttributeVisitor.

    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Avoid doing any work if nothing is changing anyway.
        if (codeAttributeEditor != null)
        {
            if (DEBUG)
            {
                System.out.println("InstructionWriter: widening instructions in "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz));
            }

            // Apply the collected expansions.
            codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);

            // Don't keep the editor around. We're assuming it won't be needed
            // very often, so we don't want to be resetting it all the time.
            codeAttributeEditor = null;
        }
    }


    // Small utility methods.

    /**
     * Remembers to place the given instruction right before the instruction
     * at the given offset.
     */
    private void insertBeforeInstruction(int instructionOffset, Instruction instruction)
    {
        ensureCodeAttributeEditor();

        // Replace the instruction.
        codeAttributeEditor.insertBeforeInstruction(instructionOffset, instruction);
    }


    /**
     * Remembers to replace the instruction at the given offset by the given
     * instruction.
     */
    private void replaceInstruction(int instructionOffset, Instruction instruction)
    {
        ensureCodeAttributeEditor();

        // Replace the instruction.
        codeAttributeEditor.replaceInstruction(instructionOffset, instruction);
    }


    /**
     * Remembers to place the given instruction right after the instruction
     * at the given offset.
     */
    private void insertAfterInstruction(int instructionOffset, Instruction instruction)
    {
        ensureCodeAttributeEditor();

        // Replace the instruction.
        codeAttributeEditor.insertAfterInstruction(instructionOffset, instruction);
    }


    /**
     * Makes sure there is a code attribute editor for the given code attribute.
     */
    private void ensureCodeAttributeEditor()
    {
        if (codeAttributeEditor == null)
        {
            codeAttributeEditor = new CodeAttributeEditor(false, true);
            codeAttributeEditor.reset(codeLength);
        }
    }
}
