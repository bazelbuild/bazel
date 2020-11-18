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
import proguard.classfile.constant.*;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This ConstantVisitor travels from any method handle constants that it visits
 * to their methodref constants, and applies a given constant visitor.
 *
 * @author Eric Lafortune
 */
public class MethodrefTraveler
extends      SimplifiedVisitor
implements   ConstantVisitor
{
    private ConstantVisitor methodrefConstantVisitor;


    /**
     * Creates a new v that will delegate to the given constant visitor.
     */
    public MethodrefTraveler(ConstantVisitor methodrefConstantVisitor)
    {
        this.methodrefConstantVisitor = methodrefConstantVisitor;
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitMethodHandleConstant(Clazz clazz, MethodHandleConstant methodHandleConstant)
    {
        clazz.constantPoolEntryAccept(methodHandleConstant.u2referenceIndex,
                                      methodrefConstantVisitor);
    }
}
