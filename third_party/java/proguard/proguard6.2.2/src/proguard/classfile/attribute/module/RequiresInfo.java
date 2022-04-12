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

import proguard.classfile.*;
import proguard.classfile.constant.visitor.ConstantVisitor;

/**
 * Representation of a Requires entry in a Module attribute.
 *
 * @author Joachim Vandersmissen
 */
public class RequiresInfo implements VisitorAccepter
{
    public int u2requiresIndex;
    public int u2requiresFlags;
    public int u2requiresVersionIndex;


    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Creates an uninitialized RequiresInfo.
     */
    public RequiresInfo()
    {
    }


    /**
     * Creates an uninitialized RequiresInfo.
     */
    public RequiresInfo(int u2requiresIndex,
                        int u2requiresFlags,
                        int u2requiresVersionIndex)
    {
        this.u2requiresIndex        = u2requiresIndex;
        this.u2requiresFlags        = u2requiresFlags;
        this.u2requiresVersionIndex = u2requiresVersionIndex;
    }


    /**
     * Applies the given constant pool visitor to the module constant of the
     * module, if any.
     */
    public void moduleAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        if (u2requiresIndex != 0)
        {
            clazz.constantPoolEntryAccept(u2requiresIndex, constantVisitor);
        }
    }

    /**
     * Applies the given constant pool visitor to the Utf8 constant of the
     * version, if any.
     */
    public void versionAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        if (u2requiresVersionIndex != 0)
        {
            clazz.constantPoolEntryAccept(u2requiresVersionIndex, constantVisitor);
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
