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
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.SimplifiedVisitor;

import java.util.Arrays;

/**
 * This AttributeVisitor accumulates specified changes to local variables, and
 * then applies these accumulated changes to the code attributes that it visits.
 *
 * @author Eric  Lafortune
 */
public class VariableEditor
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    private boolean   modified;

    private boolean[] deleted     = new boolean[ClassConstants.TYPICAL_VARIABLES_SIZE];
    private int[]     variableMap = new int[ClassConstants.TYPICAL_VARIABLES_SIZE];

    private final VariableRemapper variableRemapper = new VariableRemapper();


    /**
     * Resets the accumulated code changes.
     * @param maxLocals the length of the local variable frame that will be
     *                  edited next.
     */
    public void reset(int maxLocals)
    {
        // Try to reuse the previous array.
        if (deleted.length < maxLocals)
        {
            // Create a new array.
            deleted = new boolean[maxLocals];
        }
        else
        {
            // Reset the array.
            Arrays.fill(deleted, 0, maxLocals, false);
        }

        modified = false;
    }


    /**
     * Remembers to delete the given variable.
     * @param variableIndex the index of the variable to be deleted.
     */
    public void deleteVariable(int variableIndex)
    {
        deleted[variableIndex] = true;

        modified = true;
    }


    /**
     * Returns whether the given variable at the given offset will be deleted.
     */
    public boolean isDeleted(int instructionOffset)
    {
        return deleted[instructionOffset];
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Avoid doing any work if nothing is changing anyway.
        if (!modified)
        {
            return;
        }

        int oldMaxLocals = codeAttribute.u2maxLocals;

        // Make sure there is a sufficiently large variable map.
        if (variableMap.length < oldMaxLocals)
        {
            variableMap = new int[oldMaxLocals];
        }

        // Fill out the variable map.
        int newVariableIndex = 0;
        for (int oldVariableIndex = 0; oldVariableIndex < oldMaxLocals; oldVariableIndex++)
        {
            variableMap[oldVariableIndex] = deleted[oldVariableIndex] ?
                -1 : newVariableIndex++;
        }

        // Set the map.
        variableRemapper.setVariableMap(variableMap);

        // Remap the variables.
        variableRemapper.visitCodeAttribute(clazz, method, codeAttribute);

        // Update the length of local variable frame.
        codeAttribute.u2maxLocals = newVariableIndex;
    }
}
