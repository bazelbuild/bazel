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
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.editor.CodeAttributeEditor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This InstructionVisitor removes all nop instructions that it encounters.
 *
 * @author Eric Lafortune
 */
public class NopRemover
extends      SimplifiedVisitor
implements   InstructionVisitor
{
    private final CodeAttributeEditor codeAttributeEditor;
    private final InstructionVisitor  extraInstructionVisitor;


    /**
     * Creates a new NopRemover.
     * @param codeAttributeEditor a code editor that can be used for
     *                            accumulating changes to the code.
     */
    public NopRemover(CodeAttributeEditor codeAttributeEditor)
    {
        this(codeAttributeEditor, null);
    }


    /**
     * Creates a new NopRemover.
     * @param codeAttributeEditor     a code editor that can be used for
     *                                accumulating changes to the code.
     * @param extraInstructionVisitor an optional extra visitor for all removed
     *                                nop instructions.
     */
    public NopRemover(CodeAttributeEditor codeAttributeEditor,
                      InstructionVisitor  extraInstructionVisitor)
    {
        this.codeAttributeEditor     = codeAttributeEditor;
        this.extraInstructionVisitor = extraInstructionVisitor;
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        // Check if the instruction is a nop instruction.
        if (simpleInstruction.opcode == InstructionConstants.OP_NOP &&
            !codeAttributeEditor.isModified(offset))
        {
            codeAttributeEditor.deleteInstruction(offset);

            // Visit the instruction, if required.
            if (extraInstructionVisitor != null)
            {
                extraInstructionVisitor.visitSimpleInstruction(clazz, method, codeAttribute, offset, simpleInstruction);
            }
        }
    }
}
