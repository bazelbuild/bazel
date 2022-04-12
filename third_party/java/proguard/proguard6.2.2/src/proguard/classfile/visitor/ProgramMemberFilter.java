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


/**
 * This <code>MemberVisitor</code> delegates its visits to another given
 * <code>MemberVisitor</code>, but only when visiting members of program
 * classes.
 *
 * @author Eric Lafortune
 */
public class ProgramMemberFilter implements MemberVisitor
{
    private final MemberVisitor memberVisitor;


    /**
     * Creates a new ProgramMemberFilter.
     * @param memberVisitor     the <code>MemberVisitor</code> to which
     *                          visits will be delegated.
     */
    public ProgramMemberFilter(MemberVisitor memberVisitor)
    {
        this.memberVisitor = memberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        memberVisitor.visitProgramField(programClass, programField);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        memberVisitor.visitProgramMethod(programClass, programMethod);
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        // Don't delegate visits to library members.
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        // Don't delegate visits to library members.
    }
}
