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
package proguard.optimize.gson;

import proguard.classfile.*;
import proguard.classfile.attribute.InnerClassesInfo;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;

/**
 * Checks whether a visited class is a local or anonymous inner class.
 *
 * @author Lars Vandenbergh
 */
class      LocalOrAnonymousClassChecker
extends    SimplifiedVisitor
implements ClassVisitor,
           InnerClassesInfoVisitor
{
    private boolean localOrAnonymous;


    public boolean isLocalOrAnonymous()
    {
        return localOrAnonymous;
    }


    // Implementations for ClassVisitor.

    @Override
    public void visitAnyClass(Clazz clazz) {}


    @Override
    public void visitProgramClass(ProgramClass programClass)
    {
        localOrAnonymous = false;
        programClass.attributesAccept(new AllInnerClassesInfoVisitor(this));
    }


    // Implementations for InnerClassesInfoVisitor.

    @Override
    public void visitInnerClassesInfo(Clazz clazz, InnerClassesInfo innerClassesInfo)
    {
        if (innerClassesInfo.u2innerClassIndex == ((ProgramClass)clazz).u2thisClass)
        {
            localOrAnonymous = innerClassesInfo.u2outerClassIndex == 0 ||
                               innerClassesInfo.u2innerNameIndex  == 0;
        }
    }
}
