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
 * This Constant represents a interface method reference constant in the constant pool.
 *
 * @author Eric Lafortune
 */
public class InterfaceMethodrefConstant extends RefConstant
{
    /**
     * Creates an uninitialized InterfaceMethodrefConstant.
     */
    public InterfaceMethodrefConstant()
    {
    }


    /**
     * Creates a new InterfaceMethodrefConstant with the given name and type indices.
     * @param u2classIndex         the index   of the class in the constant pool.
     * @param u2nameAndTypeIndex   the index   of the name and type entry in the constant pool.
     * @param referencedClass      the referenced class.
     * @param referencedMember     the referenced member info.
     */
    public InterfaceMethodrefConstant(int    u2classIndex,
                                      int    u2nameAndTypeIndex,
                                      Clazz  referencedClass,
                                      Member referencedMember)
    {
        this.u2classIndex       = u2classIndex;
        this.u2nameAndTypeIndex = u2nameAndTypeIndex;
        this.referencedClass    = referencedClass;
        this.referencedMember   = referencedMember;
    }


    // Implementations for Constant.

    public int getTag()
    {
        return ClassConstants.CONSTANT_InterfaceMethodref;
    }

    public void accept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        constantVisitor.visitInterfaceMethodrefConstant(clazz, this);
    }
}
