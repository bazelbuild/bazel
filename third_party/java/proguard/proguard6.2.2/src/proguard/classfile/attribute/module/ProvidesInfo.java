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
 * Representation of a Provides entry in a Module attribute.
 *
 * @author Joachim Vandersmissen
 */
public class ProvidesInfo implements VisitorAccepter
{
    public int   u2providesIndex;
    public int   u2providesWithCount;
    public int[] u2providesWithIndex;


    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Creates an uninitialized ProvidesInfo.
     */
    public ProvidesInfo()
    {
    }


    /**
     * Creates an initialized ProvidesInfo.
     */
    public ProvidesInfo(int   u2providesIndex,
                        int   u2providesWithCount,
                        int[] u2providesWithIndex)
    {
        this.u2providesIndex     = u2providesIndex;
        this.u2providesWithCount = u2providesWithCount;
        this.u2providesWithIndex = u2providesWithIndex;
    }


    /**
     * Applies the given constant pool visitor to the class constant of the
     * provides, if any.
     */
    public void providesAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        if (u2providesIndex != 0)
        {
            clazz.constantPoolEntryAccept(u2providesIndex, constantVisitor);
        }
    }


    /**
     * Applies the given constant pool visitor to all with entries.
     */
    public void withAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        // Loop over all u2providesWithIndex entries.
        for (int index = 0; index < u2providesWithCount; index++)
        {
            clazz.constantPoolEntryAccept(u2providesWithIndex[index], constantVisitor);
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
