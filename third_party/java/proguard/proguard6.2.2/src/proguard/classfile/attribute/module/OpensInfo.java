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
 * Representation of a Opens entry in a Module attribute.
 *
 * @author Joachim Vandersmissen
 */
public class OpensInfo implements VisitorAccepter
{
    public int   u2opensIndex;
    public int   u2opensFlags;
    public int   u2opensToCount;
    public int[] u2opensToIndex;


    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Creates an uninitialized OpensInfo.
     */
    public OpensInfo()
    {
    }


    /**
     * Creates an initialized OpensInfo.
     */
    public OpensInfo(int   u2opensIndex,
                     int   u2opensFlags,
                     int   u2opensToCount,
                     int[] u2opensToIndex)
    {
        this.u2opensIndex   = u2opensIndex;
        this.u2opensFlags   = u2opensFlags;
        this.u2opensToCount = u2opensToCount;
        this.u2opensToIndex = u2opensToIndex;
    }


    /**
     * Applies the given constant pool visitor to the package constant of the
     * package, if any.
     */
    public void packageAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        if (u2opensIndex != 0)
        {
            clazz.constantPoolEntryAccept(u2opensIndex, constantVisitor);
        }
    }


    /**
     * Applies the given constant pool visitor to all targets.
     */
    public void targetsAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        // Loop over all targets.
        for (int index = 0; index < u2opensToCount; index++)
        {
            clazz.constantPoolEntryAccept(u2opensToIndex[index], constantVisitor);
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
