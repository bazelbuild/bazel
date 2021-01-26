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
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;

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
    private int[] bootstrapMethodIndexMap;

    // Ignore (skip) lingering InvokeDynamic constants that
    // refer to removed bootstrap methods.
    private final boolean ignoreDanglingConstants;

    public BootstrapMethodRemapper()
    {
        this(false);
    }

    public BootstrapMethodRemapper(boolean ignoreDanglingConstants)
    {
        this.ignoreDanglingConstants = ignoreDanglingConstants;
    }


    /**
     * Sets the given mapping of old constant pool entry indexes to their new
     * indexes.
     */
    public void setBootstrapMethodIndexMap(int[] bootstrapMethodIndexMap)
    {
        this.bootstrapMethodIndexMap = bootstrapMethodIndexMap;
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitDynamicConstant(Clazz clazz, DynamicConstant dynamicConstant)
    {
        dynamicConstant.u2bootstrapMethodAttributeIndex =
            remapConstantIndex(dynamicConstant.u2bootstrapMethodAttributeIndex);
    }


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        invokeDynamicConstant.u2bootstrapMethodAttributeIndex =
            remapConstantIndex(invokeDynamicConstant.u2bootstrapMethodAttributeIndex);
    }


    // Small utility methods.

    /**
     * Returns the latest bootstrap method index of the entry at the
     * given index.
     */
    private int remapConstantIndex(int constantIndex)
    {
        int remappedConstantIndex = bootstrapMethodIndexMap[constantIndex];
        if (remappedConstantIndex < 0)
        {
            if (ignoreDanglingConstants)
            {
                return constantIndex;
            }
            else
            {
                throw new IllegalArgumentException("Can't remap bootstrap method index ["+constantIndex+"]");
            }
        }

        return remappedConstantIndex;
    }
}
