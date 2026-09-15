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
import proguard.classfile.visitor.ClassVisitor;

/**
 * This Attribute represents a main class attribute.
 *
 * @author Joachim Vandersmissen
 */
public class ModuleMainClassAttribute extends Attribute
{
    public int u2mainClass;


    /**
     * Creates an uninitialized ModuleMainClassAttribute.
     */
    public ModuleMainClassAttribute()
    {
    }


    /**
     * Creates an initialized ModuleMainClassAttribute.
     */
    public ModuleMainClassAttribute(int u2attributeNameIndex, int u2mainClass)
    {
        super(u2attributeNameIndex);
        this.u2mainClass = u2mainClass;
    }


    // Implementations for Attribute.

    public void accept(Clazz clazz, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitModuleMainClassAttribute(clazz, this);
    }


    /**
     * Applies the given constant pool visitor to the class constant of the
     * main class, if any.
     */
    public void mainClassAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        if (u2mainClass != 0)
        {
            clazz.constantPoolEntryAccept(u2mainClass, constantVisitor);
        }
    }
}
