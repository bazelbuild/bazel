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
 * This Constant represents a package constant in the constant pool.
 *
 * @author Joachim Vandersmissen
 */
public class PackageConstant extends Constant
{
    public int u2nameIndex;


    /**
     * Creates an uninitialized PackageConstant.
     */
    public PackageConstant()
    {
    }


    /**
     * Creates a new PackageConstant with the given name index.
     * @param u2nameIndex the index of the name in the constant pool.
     */
    public PackageConstant(int u2nameIndex)
    {
        this.u2nameIndex = u2nameIndex;
    }


    /**
     * Returns the name.
     */
    public String getName(Clazz clazz)
    {
        return clazz.getString(u2nameIndex);
    }


    // Implementations for Constant.

    public int getTag()
    {
        return ClassConstants.CONSTANT_Package;
    }

    public void accept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        constantVisitor.visitPackageConstant(clazz, this);
    }
}
