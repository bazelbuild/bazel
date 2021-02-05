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
package proguard.classfile.attribute.preverification;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.attribute.preverification.visitor.StackMapFrameVisitor;

/**
 * This StackMapFrame represents an "chop frame".
 *
 * @author Eric Lafortune
 */
public class LessZeroFrame extends StackMapFrame
{
    public int choppedVariablesCount;


    /**
     * Creates an uninitialized LessZeroFrame.
     */
    public LessZeroFrame()
    {
    }


    /**
     * Creates a LessZeroFrame with the given tag.
     */
    public LessZeroFrame(int tag)
    {
        choppedVariablesCount = LESS_ZERO_FRAME + 3 - tag;
    }


    /**
     * Creates a LessZeroFrame with the given number of chopped variables.
     */
    public LessZeroFrame(byte choppedVariablesCount)
    {
        this.choppedVariablesCount = (int)choppedVariablesCount;
    }


    // Implementations for StackMapFrame.

    public int getTag()
    {
        return LESS_ZERO_FRAME + 3 - choppedVariablesCount;
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, StackMapFrameVisitor stackMapFrameVisitor)
    {
        stackMapFrameVisitor.visitLessZeroFrame(clazz, method, codeAttribute, offset, this);
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (!super.equals(object))
        {
            return false;
        }

        LessZeroFrame other = (LessZeroFrame)object;

        return this.u2offsetDelta == other.u2offsetDelta &&
               this.choppedVariablesCount != other.choppedVariablesCount;
    }


    public int hashCode()
    {
        return super.hashCode() ^ choppedVariablesCount;
    }


    public String toString()
    {
        return super.toString()+"Var: (chopped "+choppedVariablesCount+"), Stack: (empty)";
    }
}
