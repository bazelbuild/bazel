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
 * This Constant represents a integer constant in the constant pool.
 *
 * @author Eric Lafortune
 */
public class IntegerConstant extends Constant
{
    public int u4value;


    /**
     * Creates an uninitialized IntegerConstant.
     */
    public IntegerConstant()
    {
    }


    /**
     * Creates a new IntegerConstant with the given integer value.
     */
    public IntegerConstant(int value)
    {
        u4value = value;
    }


    /**
     * Returns the integer value of this IntegerConstant.
     */
    public int getValue()
    {
        return u4value;
    }


    /**
     * Sets the integer value of this IntegerConstant.
     */
    public void setValue(int value)
    {
        u4value = value;
    }


    // Implementations for Constant.

    public int getTag()
    {
        return ClassConstants.CONSTANT_Integer;
    }

    public void accept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        constantVisitor.visitIntegerConstant(clazz, this);
    }
}
