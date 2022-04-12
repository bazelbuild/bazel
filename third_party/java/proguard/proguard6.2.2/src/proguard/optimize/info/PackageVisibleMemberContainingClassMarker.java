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
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

/**
 * This ClassVisitor marks all classes that contain package visible members.
 *
 * @author Eric Lafortune
 */
public class PackageVisibleMemberContainingClassMarker
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor
{
    // Implementations for ClassVisitor.

    public void visitAnyClass(Clazz clazz)
    {
        // Check the class itself.
        if ((clazz.getAccessFlags() & ClassConstants.ACC_PUBLIC) == 0)
        {
            setPackageVisibleMembers(clazz);
        }
        else
        {
            // Check the members.
            clazz.fieldsAccept(this);
            clazz.methodsAccept(this);
        }
    }


    // Implementations for MemberVisitor.

    public void visitAnyMember(Clazz clazz, Member member)
    {
        if ((member.getAccessFlags() &
             (ClassConstants.ACC_PRIVATE |
              ClassConstants.ACC_PUBLIC)) == 0)
        {
            setPackageVisibleMembers(clazz);
        }
    }


    // Small utility methods.

    private static void setPackageVisibleMembers(Clazz clazz)
    {
        ProgramClassOptimizationInfo.getProgramClassOptimizationInfo(clazz).setContainsPackageVisibleMembers();
    }


    public static boolean containsPackageVisibleMembers(Clazz clazz)
    {
        return ClassOptimizationInfo.getClassOptimizationInfo(clazz).containsPackageVisibleMembers();
    }
}
