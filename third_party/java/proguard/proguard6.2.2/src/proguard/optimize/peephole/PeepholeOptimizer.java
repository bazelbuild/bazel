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
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.editor.CodeAttributeEditor;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This AttributeVisitor sets up and applies the peephole optimizations of its
 * instruction visitor. The instruction visitor should be using the same
 * (optional) branch target finder and code attribute editor.
 *
 * @author Eric Lafortune
 */
public class PeepholeOptimizer
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    private final BranchTargetFinder  branchTargetFinder;
    private final CodeAttributeEditor codeAttributeEditor;
    private final InstructionVisitor  instructionVisitor;


    /**
     * Creates a new PeepholeOptimizer.
     * @param codeAttributeEditor the code attribute editor that will be reset
     *                            and then executed.
     * @param instructionVisitor  the instruction visitor that performs
     *                            peephole optimizations using the above code
     *                            attribute editor.
     */
    public PeepholeOptimizer(CodeAttributeEditor codeAttributeEditor,
                             InstructionVisitor  instructionVisitor)
    {
        this(null, codeAttributeEditor, instructionVisitor);
    }


    /**
     * Creates a new PeepholeOptimizer.
     * @param branchTargetFinder  branch target finder that will be initialized
     *                            to indicate branch targets in the visited code.
     * @param codeAttributeEditor the code attribute editor that will be reset
     *                            and then executed.
     * @param instructionVisitor  the instruction visitor that performs
     *                            peephole optimizations using the above code
     *                            attribute editor.
     */
    public PeepholeOptimizer(BranchTargetFinder  branchTargetFinder,
                             CodeAttributeEditor codeAttributeEditor,
                             InstructionVisitor  instructionVisitor)
    {
        this.branchTargetFinder  = branchTargetFinder;
        this.codeAttributeEditor = codeAttributeEditor;
        this.instructionVisitor  = instructionVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        if (branchTargetFinder != null)
        {
            // Set up the branch target finder.
            branchTargetFinder.visitCodeAttribute(clazz, method, codeAttribute);
        }

        // Set up the code attribute editor.
        codeAttributeEditor.reset(codeAttribute.u4codeLength);

        // Find the peephole optimizations.
        codeAttribute.instructionsAccept(clazz, method, instructionVisitor);

        // Apply the peephole optimizations.
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
    }
}
