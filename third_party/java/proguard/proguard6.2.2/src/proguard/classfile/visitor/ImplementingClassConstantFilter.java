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

import proguard.classfile.Clazz;
import proguard.classfile.constant.ClassConstant;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This <code>ConstantVisitor</code> delegates its visits to class constants
 * to another given <code>ConstantVisitor</code>, except for classes that
 * are extended or implemented by a given class. This exception includes the
 * class itself.
 *
 * @author Eric Lafortune
 */
public class ImplementingClassConstantFilter
extends      SimplifiedVisitor
implements   ConstantVisitor
{
    private final Clazz           implementingClass;
    private final ConstantVisitor constantVisitor;


    /**
     * Creates a new ImplementingClassConstantFilter.
     * @param implementingClass the class whose superclasses and interfaces will
     *                          not be visited.
     * @param constantVisitor   the <code>ConstantVisitor</code> to which visits
     *                          will be delegated.
     */
    public ImplementingClassConstantFilter(Clazz           implementingClass,
                                           ConstantVisitor constantVisitor)
    {
        this.implementingClass = implementingClass;
        this.constantVisitor   = constantVisitor;
    }


    // Implementations for ConstantVisitor.

    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        Clazz referencedClass = classConstant.referencedClass;
        if (referencedClass == null ||
            !implementingClass.extendsOrImplements(referencedClass))
        {
            constantVisitor.visitClassConstant(clazz, classConstant);
        }
    }
}