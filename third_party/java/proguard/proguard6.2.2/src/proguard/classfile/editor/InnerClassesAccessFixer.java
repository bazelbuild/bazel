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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.attribute.InnerClassesInfo;
import proguard.classfile.attribute.visitor.InnerClassesInfoVisitor;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This InnerClassesInfoVisitor fixes the inner class access flags of the
 * inner classes information that it visits.
 *
 * @author Eric Lafortune
 */
public class InnerClassesAccessFixer
extends      SimplifiedVisitor
implements   InnerClassesInfoVisitor,
             ConstantVisitor,
             ClassVisitor
{
    private int innerClassAccessFlags;


    // Implementations for InnerClassesInfoVisitor.

    public void visitInnerClassesInfo(Clazz clazz, InnerClassesInfo innerClassesInfo)
    {
        // The current access flags are the default.
        innerClassAccessFlags = innerClassesInfo.u2innerClassAccessFlags;

        // See if we can find new access flags.
        innerClassesInfo.innerClassConstantAccept(clazz, this);

        // Update the access flags.
        innerClassesInfo.u2innerClassAccessFlags = innerClassAccessFlags;
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        classConstant.referencedClassAccept(this);
    }


    // Implementations for ClassVisitor.

    public void visitLibraryClass(LibraryClass libraryClass) {}


    public void visitProgramClass(ProgramClass programClass)
    {
        innerClassAccessFlags =
            AccessUtil.replaceAccessFlags(innerClassAccessFlags,
                                          programClass.u2accessFlags);
    }
}