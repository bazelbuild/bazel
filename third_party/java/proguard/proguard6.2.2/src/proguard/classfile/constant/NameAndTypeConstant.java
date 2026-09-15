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
package proguard.classfile.constant;

import proguard.classfile.*;
import proguard.classfile.constant.visitor.ConstantVisitor;

/**
 * This Constant represents a name and type constant in the constant pool.
 *
 * @author Eric Lafortune
 */
public class NameAndTypeConstant extends Constant
{
    public int u2nameIndex;
    public int u2descriptorIndex;


    /**
     * Creates an uninitialized NameAndTypeConstant.
     */
    public NameAndTypeConstant()
    {
    }


    /**
     * Creates a new NameAndTypeConstant with the given name and type indices.
     * @param u2nameIndex       the index of the name in the constant pool.
     * @param u2descriptorIndex the index of the descriptor in the constant
     *                          pool.
     */
    public NameAndTypeConstant(int u2nameIndex,
                               int u2descriptorIndex)
    {
        this.u2nameIndex       = u2nameIndex;
        this.u2descriptorIndex = u2descriptorIndex;
    }


    /**
     * Returns the name index.
     */
    protected int getNameIndex()
    {
        return u2nameIndex;
    }

    /**
     * Sets the name index.
     */
    protected void setNameIndex(int index)
    {
        u2nameIndex = index;
    }

    /**
     * Returns the descriptor index.
     */
    protected int getDescriptorIndex()
    {
        return u2descriptorIndex;
    }

    /**
     * Sets the descriptor index.
     */
    protected void setDescriptorIndex(int index)
    {
        u2descriptorIndex = index;
    }

    /**
     * Returns the name.
     */
    public String getName(Clazz clazz)
    {
        return clazz.getString(u2nameIndex);
    }

    /**
     * Returns the type.
     */
    public String getType(Clazz clazz)
    {
        return clazz.getString(u2descriptorIndex);
    }


    // Implementations for Constant.

    public int getTag()
    {
        return ClassConstants.CONSTANT_NameAndType;
    }

    public void accept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        constantVisitor.visitNameAndTypeConstant(clazz, this);
    }
}
