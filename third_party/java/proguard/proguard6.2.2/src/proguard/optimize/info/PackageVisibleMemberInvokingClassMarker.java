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
package proguard.optimize.info;

import proguard.classfile.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

/**
 * This ConstantVisitor marks all classes that refer to package visible classes
 * or class members.
 *
 * @author Eric Lafortune
 */
public class PackageVisibleMemberInvokingClassMarker
extends      SimplifiedVisitor
implements   ConstantVisitor,
             ClassVisitor,
             MemberVisitor
{
    private Clazz referencingClass;


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        // Check the referenced class and class member, if any.
        if (stringConstant.referencedClass != clazz)
        {
            referencingClass = clazz;

            stringConstant.referencedClassAccept(this);
            stringConstant.referencedMemberAccept(this);
        }
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        // Check the referenced class and class member.
        if (refConstant.referencedClass != clazz)
        {
            referencingClass = clazz;

            refConstant.referencedClassAccept(this);
            refConstant.referencedMemberAccept(this);
        }
    }


    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Check the referenced class.
        if (classConstant.referencedClass != clazz)
        {
            referencingClass = clazz;

            classConstant.referencedClassAccept(this);
        }
    }


    // Implementations for ClassVisitor.

    public void visitAnyClass(Clazz clazz)
    {
        if ((clazz.getAccessFlags() &
             ClassConstants.ACC_PUBLIC) == 0)
        {
            setInvokesPackageVisibleMembers(referencingClass);
        }
    }


    // Implementations for MemberVisitor.

    public void visitAnyMember(Clazz clazz, Member member)
    {
        if ((member.getAccessFlags() &
             (ClassConstants.ACC_PUBLIC |
              ClassConstants.ACC_PRIVATE)) == 0)
        {
            setInvokesPackageVisibleMembers(referencingClass);
        }
    }


    // Small utility methods.

    private static void setInvokesPackageVisibleMembers(Clazz clazz)
    {
        ProgramClassOptimizationInfo.getProgramClassOptimizationInfo(clazz).setInvokesPackageVisibleMembers();
    }


    public static boolean invokesPackageVisibleMembers(Clazz clazz)
    {
        return ClassOptimizationInfo.getClassOptimizationInfo(clazz).invokesPackageVisibleMembers();
    }
}
