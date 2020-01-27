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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.attribute.preverification.*;
import proguard.classfile.attribute.preverification.visitor.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

/**
 * This ConstantVisitor remaps all possible indices of bootstrap methods
 * of the constants that it visits, based on a given index map.
 *
 * @author Eric Lafortune
 */
public class BootstrapMethodRemapper
extends      SimplifiedVisitor
implements   ConstantVisitor
{
    private int[] constantIndexMap;


    /**
     * Sets the given mapping of old constant pool entry indexes to their new
     * indexes.
     */
    public void setConstantIndexMap(int[] constantIndexMap)
    {
        this.constantIndexMap = constantIndexMap;
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        invokeDynamicConstant.u2bootstrapMethodAttributeIndex =
            remapConstantIndex(invokeDynamicConstant.u2bootstrapMethodAttributeIndex);
    }


    // Small utility methods.

    /**
     * Returns the new bootstrap method index of the entry at the
     * given index.
     */
    private int remapConstantIndex(int constantIndex)
    {
        int remappedConstantIndex = constantIndexMap[constantIndex];
        if (remappedConstantIndex < 0)
        {
            throw new IllegalArgumentException("Can't remap constant index ["+constantIndex+"]");
        }

        return remappedConstantIndex;
    }
}
