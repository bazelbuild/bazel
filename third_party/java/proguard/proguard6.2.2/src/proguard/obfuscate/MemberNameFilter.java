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
package proguard.obfuscate;

import proguard.classfile.*;
import proguard.classfile.visitor.MemberVisitor;

/**
 * This <code>MemberVisitor</code> delegates its visits to another given
 * <code>MemberVisitor</code>, but only when the visited member has a new name.
 * Constructors are judged based on the class name.
 *
 * @see ClassObfuscator
 * @see MemberObfuscator
 *
 * @author Eric Lafortune
 */
public class MemberNameFilter implements MemberVisitor
{
    private final MemberVisitor memberVisitor;


    /**
     * Creates a new MemberNameFilter.
     * @param memberVisitor the <code>MemberVisitor</code> to which
     *                      visits will be delegated.
     */
    public MemberNameFilter(MemberVisitor memberVisitor)
    {
        this.memberVisitor = memberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        if (hasName(programField))
        {
            memberVisitor.visitProgramField(programClass, programField);
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if (hasName(programClass, programMethod))
        {
            memberVisitor.visitProgramMethod(programClass, programMethod);
        }
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        if (hasName(libraryField))
        {
            memberVisitor.visitLibraryField(libraryClass, libraryField);
        }
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        if (hasName(libraryClass, libraryMethod))
        {
            memberVisitor.visitLibraryMethod(libraryClass, libraryMethod);
        }
    }


    // Small utility methods.

    /**
     * Returns whether the given class has a new name.
     */
    private boolean hasName(Clazz clazz)
    {
        return ClassObfuscator.newClassName(clazz) != null;
    }


    /**
     * Returns whether the given method has a new name.
     */
    private boolean hasName(Clazz clazz, Method method)
    {
        return
            hasName(method) ||
            (hasName(clazz) &&
             method.getName(clazz).equals(ClassConstants.METHOD_NAME_INIT));
    }


    /**
     * Returns whether the given class member has a new name.
     */
    private boolean hasName(Member member)
    {
        return MemberObfuscator.newMemberName(member) != null;
    }
}
