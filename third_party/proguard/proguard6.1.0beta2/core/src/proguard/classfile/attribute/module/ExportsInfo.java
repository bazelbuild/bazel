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
 * Representation of a Exports entry in a Module attribute.
 *
 * @author Joachim Vandersmissen
 */
public class ExportsInfo implements VisitorAccepter
{
    public int   u2exportsIndex;
    public int   u2exportsFlags;
    public int   u2exportsToCount;
    public int[] u2exportsToIndex;


    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Creates an uninitialized ExportsInfo.
     */
    public ExportsInfo()
    {
    }


    /**
     * Creates an initialized ExportsInfo.
     */
    public ExportsInfo(int   u2exportsIndex,
                       int   u2exportsFlags,
                       int   u2exportsToCount,
                       int[] u2exportsToIndex)
    {
        this.u2exportsIndex   = u2exportsIndex;
        this.u2exportsFlags   = u2exportsFlags;
        this.u2exportsToCount = u2exportsToCount;
        this.u2exportsToIndex = u2exportsToIndex;
    }


    /**
     * Applies the given constant pool visitor to the package constant of the
     * package, if any.
     */
    public void packageAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        if (u2exportsIndex != 0)
        {
            clazz.constantPoolEntryAccept(u2exportsIndex, constantVisitor);
        }
    }


    /**
     * Applies the given constant pool visitor to all exportsToIndex.
     */
    public void exportsToIndexAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        // Loop over all exportsToIndex.
        for (int index = 0; index < u2exportsToCount; index++)
        {
            clazz.constantPoolEntryAccept(u2exportsToIndex[index], constantVisitor);
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
