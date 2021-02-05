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
 * This StackMapFrame represents a "same frame" or a "same frame extended".
 *
 * @author Eric Lafortune
 * @noinspection PointlessArithmeticExpression
 */
public class SameZeroFrame extends StackMapFrame
{
    /**
     * Creates an uninitialized SameZeroFrame.
     */
    public SameZeroFrame()
    {
    }


    /**
     * Creates a SameZeroFrame with the given tag.
     */
    public SameZeroFrame(int tag)
    {
        u2offsetDelta = tag - SAME_ZERO_FRAME;
    }


    // Implementations for StackMapFrame.

    public int getTag()
    {
        return u2offsetDelta < 64 ?
            SAME_ZERO_FRAME + u2offsetDelta :
            SAME_ZERO_FRAME_EXTENDED;
    }


    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, StackMapFrameVisitor stackMapFrameVisitor)
    {
        stackMapFrameVisitor.visitSameZeroFrame(clazz, method, codeAttribute, offset, this);
    }


    // Implementations for Object.

    public String toString()
    {
        return super.toString()+"Var: ..., Stack: (empty)";
    }
}
