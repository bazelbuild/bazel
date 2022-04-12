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
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.*;

/**
 * This <code>MemberVisitor</code> delegates its visits to another given
 * <code>MemberVisitor</code>, but only when the visited member is accessible
 * from the given referencing class.
 *
 * @author Eric Lafortune
 */
public class MemberClassAccessFilter
implements   MemberVisitor
{
    private final NestHostFinder nestHostFinder = new NestHostFinder();
    private final Clazz          referencingClass;
    private final String         referencingNestHostClassName;
    private final MemberVisitor  memberVisitor;



    /**
     * Creates a new MemberAccessFilter.
     * @param referencingClass the class that is accessing the member.
     * @param memberVisitor    the <code>MemberVisitor</code> to which visits
     *                         will be delegated.
     */
    public MemberClassAccessFilter(Clazz         referencingClass,
                                   MemberVisitor memberVisitor)
    {
        this.referencingClass             = referencingClass;
        this.referencingNestHostClassName = nestHostFinder.findNestHostClassName(referencingClass);
        this.memberVisitor                = memberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        if (accepted(programClass, programField.getAccessFlags()))
        {
            memberVisitor.visitProgramField(programClass, programField);
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if (accepted(programClass, programMethod.getAccessFlags()))
        {
            memberVisitor.visitProgramMethod(programClass, programMethod);
        }
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        if (accepted(libraryClass, libraryField.getAccessFlags()))
        {
            memberVisitor.visitLibraryField(libraryClass, libraryField);
        }
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        if (accepted(libraryClass, libraryMethod.getAccessFlags()))
        {
            memberVisitor.visitLibraryMethod(libraryClass, libraryMethod);
        }
    }


    // Small utility methods.

    private boolean accepted(Clazz clazz, int memberAccessFlags)
    {
        int accessLevel = AccessUtil.accessLevel(memberAccessFlags);

        return
            (accessLevel >= AccessUtil.PUBLIC                                                               ) ||
            (accessLevel >= AccessUtil.PRIVATE         && nestHostFinder.inSameNest(referencingClass, clazz)) ||
            (accessLevel >= AccessUtil.PACKAGE_VISIBLE && (ClassUtil.internalPackageName(referencingClass.getName()).equals(
                                                           ClassUtil.internalPackageName(clazz.getName()))) ) ||
            (accessLevel >= AccessUtil.PROTECTED       && (referencingClass.extends_(clazz) ||
                                                           referencingClass.extendsOrImplements(clazz))     );
    }
}
