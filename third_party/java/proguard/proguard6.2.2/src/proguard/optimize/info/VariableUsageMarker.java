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
package proguard.optimize.info;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;

import java.util.Arrays;

/**
 * This AttributeVisitor marks the local variables that are used in the code
 * attributes that it visits.
 *
 * @author Eric Lafortune
 */
public class VariableUsageMarker
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor
{
    private boolean[] variableUsed = new boolean[ClassConstants.TYPICAL_VARIABLES_SIZE];


    /**
     * Returns whether the given variable has been marked as being used.
     */
    public boolean isVariableUsed(int variableIndex)
    {
        return variableUsed[variableIndex];
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        int maxLocals = codeAttribute.u2maxLocals;

        // Try to reuse the previous array.
        if (variableUsed.length < maxLocals)
        {
            // Create a new array.
            variableUsed = new boolean[maxLocals];
        }
        else
        {
            // Reset the array.
            Arrays.fill(variableUsed, 0, maxLocals, false);
        }

        codeAttribute.instructionsAccept(clazz, method, this);
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        // Mark the variable.
        variableUsed[variableInstruction.variableIndex] = true;

        // Account for Category 2 instructions, which take up two entries.
        if (variableInstruction.stackPopCount(clazz)  == 2 ||
            variableInstruction.stackPushCount(clazz) == 2)
        {
            variableUsed[variableInstruction.variableIndex + 1] = true;
        }
    }
}
