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

import proguard.classfile.VisitorAccepter;

/**
 * Representation of an Exception table entry.
 *
 * @author Eric Lafortune
 */
public class ExceptionInfo implements VisitorAccepter
{
    public int u2startPC;
    public int u2endPC;
    public int u2handlerPC;
    public int u2catchType;

    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Creates an uninitialized ExceptionInfo.
     */
    public ExceptionInfo()
    {
    }


    /**
     * Creates an initialized ExceptionInfo.
     */
    public ExceptionInfo(int u2startPC,
                         int u2endPC,
                         int u2handlerPC,
                         int u2catchType)
    {
        this.u2startPC   = u2startPC;
        this.u2endPC     = u2endPC;
        this.u2handlerPC = u2handlerPC;
        this.u2catchType = u2catchType;
    }


    /**
     * Returns whether the exception's try block contains the instruction at the
     * given offset.
     */
    public boolean isApplicable(int instructionOffset)
    {
        return instructionOffset >= u2startPC &&
               instructionOffset <  u2endPC;
    }


    /**
     * Returns whether the exception's try block overlaps with the specified
     * block of instructions.
     */
    public boolean isApplicable(int startOffset, int endOffset)
    {
        return u2startPC < endOffset &&
               u2endPC   > startOffset;
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
