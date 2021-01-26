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
import proguard.classfile.attribute.BootstrapMethodInfo;
import proguard.classfile.attribute.visitor.BootstrapMethodInfoVisitor;

/**
 * This BootstrapMethodInfoVisitor lets a given ConstantVisitor visit all
 * constant pool entries of the bootstrap methods it visits.
 *
 * @author Eric Lafortune
 */
public class AllBootstrapMethodArgumentVisitor
implements   BootstrapMethodInfoVisitor
{
    private ConstantVisitor constantVisitor;

    /**
     * Creates a new AllBootstrapMethodArgumentVisitor that will delegate to
     * the given constant visitor.
     */
    public AllBootstrapMethodArgumentVisitor(ConstantVisitor constantVisitor)
    {
        this.constantVisitor = constantVisitor;
    }


    // Implementations for BootstrapMethodInfoVisitor.

    public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
    {
        bootstrapMethodInfo.methodArgumentsAccept(clazz, constantVisitor);
    }
}
