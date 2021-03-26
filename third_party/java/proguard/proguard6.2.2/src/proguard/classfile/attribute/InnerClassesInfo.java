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
package proguard.classfile.attribute;

import proguard.classfile.*;
import proguard.classfile.constant.visitor.ConstantVisitor;

/**
 * Representation of an Inner Classes table entry.
 *
 * @author Eric Lafortune
 */
public class InnerClassesInfo implements VisitorAccepter
{
    public int u2innerClassIndex;
    public int u2outerClassIndex;
    public int u2innerNameIndex;
    public int u2innerClassAccessFlags;

    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Applies the given constant pool visitor to the class constant of the
     * inner class, if any.
     */
    public void innerClassConstantAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        if (u2innerClassIndex != 0)
        {
            clazz.constantPoolEntryAccept(u2innerClassIndex, constantVisitor);
        }
    }


    /**
     * Applies the given constant pool visitor to the class constant of the
     * outer class, if any.
     */
    public void outerClassConstantAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        if (u2outerClassIndex != 0)
        {
            clazz.constantPoolEntryAccept(u2outerClassIndex, constantVisitor);
        }
    }


    /**
     * Applies the given constant pool visitor to the Utf8 constant of the
     * inner name, if any.
     */
    public void innerNameConstantAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        if (u2innerNameIndex != 0)
        {
            clazz.constantPoolEntryAccept(u2innerNameIndex, constantVisitor);
        }
    }


    // Implementations for VisitorAccepter.

    public Object getVisitorInfo()
    {
        return visitorInfo;
    }

    public void setVisitorInfo(Object visitorInfo)
    {
        this.visitorInfo = visitorInfo;
    }
}
