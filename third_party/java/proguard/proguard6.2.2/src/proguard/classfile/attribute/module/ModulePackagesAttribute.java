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
package proguard.classfile.attribute.module;

import proguard.classfile.Clazz;
import proguard.classfile.attribute.Attribute;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.constant.visitor.ConstantVisitor;

/**
 * This Attribute represents a module packages attribute.
 *
 * @author Joachim Vandersmissen
 */
public class ModulePackagesAttribute extends Attribute
{
    public int   u2packagesCount;
    public int[] u2packages;


    /**
     * Creates an uninitialized ModulePackagesAttribute.
     */
    public ModulePackagesAttribute()
    {
    }


    /**
     * Creates an initialized ModulePackagesAttribute.
     */
    public ModulePackagesAttribute(int   u2attributeNameIndex,
                                   int   u2packagesCount,
                                   int[] u2packages)
    {
        super(u2attributeNameIndex);
        this.u2packagesCount = u2packagesCount;
        this.u2packages      = u2packages;
    }


    // Implementations for Attribute.

    public void accept(Clazz clazz, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitModulePackagesAttribute(clazz, this);
    }


    /**
     * Applies the given constant pool visitor to all packages.
     */
    public void packagesAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        // Loop over all packages.
        for (int index = 0; index < u2packagesCount; index++)
        {
            clazz.constantPoolEntryAccept(u2packages[index], constantVisitor);
        }
    }
}
