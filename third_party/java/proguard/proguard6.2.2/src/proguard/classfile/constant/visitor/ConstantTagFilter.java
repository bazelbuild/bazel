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
package proguard.classfile.constant.visitor;

import proguard.classfile.Clazz;
import proguard.classfile.constant.Constant;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This <code>ConstantVisitor</code> delegates its visits to one or more
 * specified types of constants.
 *
 * @author Eric Lafortune
 */
public class ConstantTagFilter
extends      SimplifiedVisitor
implements   ConstantVisitor
{
    private final int             constantTagMask;
    private final ConstantVisitor constantVisitor;


    /**
     * Creates a new ConstantTagFilter.
     * @param constantTag     the type of constants for which visits will be
     *                        delegated.
     * @param constantVisitor the <code>ConstantVisitor</code> to which visits
     *                        will be delegated.
     */
    public ConstantTagFilter(int             constantTag,
                             ConstantVisitor constantVisitor)
    {
        this.constantTagMask = 1 << constantTag;
        this.constantVisitor = constantVisitor;
    }


    /**
     * Creates a new ConstantTagFilter.
     * @param constantTags    the types of constants for which visits will be
     *                        delegated.
     * @param constantVisitor the <code>ConstantVisitor</code> to which visits
     *                        will be delegated.
     */
    public ConstantTagFilter(int[]           constantTags,
                             ConstantVisitor constantVisitor)
    {
        int constantTagMask = 0;
        for (int index = 0; index < constantTags.length; index++)
        {
            constantTagMask |= 1 << constantTags[index];
        }

        this.constantTagMask = constantTagMask;
        this.constantVisitor = constantVisitor;
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant)
    {
        if (((1 << constant.getTag()) & constantTagMask) != 0)
        {
            constant.accept(clazz, constantVisitor);
        }
    }
}