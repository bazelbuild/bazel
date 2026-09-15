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
package proguard.classfile.visitor;

import proguard.classfile.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.util.ArrayUtil;


/**
 * This ConstantVisitor delegates all visits to each ConstantVisitor in a given list.
 *
 * @author Johan Leys
 */
public class MultiConstantVisitor
extends      SimplifiedVisitor
implements   ConstantVisitor
{
    private ConstantVisitor[] constantVisitors;
    private int               constantVisitorCount;


    public MultiConstantVisitor()
    {
        this.constantVisitors = new ConstantVisitor[16];
    }


    public MultiConstantVisitor(ConstantVisitor... constantVisitors)
    {
        this.constantVisitors     = constantVisitors;
        this.constantVisitorCount = this.constantVisitors.length;
    }


    public void addClassVisitor(ConstantVisitor constantVisitor)
    {
        constantVisitors =
            ArrayUtil.add(constantVisitors,
                          constantVisitorCount++,
                          constantVisitor);
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant)
    {
        for (int index = 0; index < constantVisitorCount; index++)
        {
            constant.accept(clazz, constantVisitors[index]);
        }
    }
}
