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
 * Representation of a bootstrap method.
 *
 * @author Eric Lafortune
 */
public class BootstrapMethodInfo implements VisitorAccepter
{
    public int   u2methodHandleIndex;
    public int   u2methodArgumentCount;
    public int[] u2methodArguments;

    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Creates an uninitialized BootstrapMethodInfo.
     */
    public BootstrapMethodInfo()
    {
    }


    /**
     * Creates an initialized BootstrapMethodInfo.
     */
    public BootstrapMethodInfo(int   u2methodHandleIndex,
                               int   u2methodArgumentCount,
                               int[] u2methodArguments)
    {
        this.u2methodHandleIndex   = u2methodHandleIndex;
        this.u2methodArgumentCount = u2methodArgumentCount;
        this.u2methodArguments     = u2methodArguments;
    }


    /**
     * Applies the given constant pool visitor to the method handle of the
     * bootstrap method.
     */
    public void methodHandleAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        clazz.constantPoolEntryAccept(u2methodHandleIndex,
                                      constantVisitor);
    }


    /**
     * Applies the given constant pool visitor to the argument constants of the
     * bootstrap method.
     */
    public void methodArgumentsAccept(Clazz clazz, ConstantVisitor constantVisitor)
    {
        for (int index = 0; index < u2methodArgumentCount; index++)
        {
            clazz.constantPoolEntryAccept(u2methodArguments[index],
                                          constantVisitor);
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
