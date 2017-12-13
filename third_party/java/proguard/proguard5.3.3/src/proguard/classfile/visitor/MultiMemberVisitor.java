/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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
 * This MemberVisitor delegates all visits to each MemberVisitor
 * in a given list.
 *
 * @author Eric Lafortune
 */
public class MultiMemberVisitor implements MemberVisitor
{
    private static final int ARRAY_SIZE_INCREMENT = 5;

    private MemberVisitor[] memberVisitors;
    private int             memberVisitorCount;


    public MultiMemberVisitor()
    {
    }


    public MultiMemberVisitor(MemberVisitor[] memberVisitors)
    {
        this.memberVisitors     = memberVisitors;
        this.memberVisitorCount = memberVisitors.length;
    }


    public void addMemberVisitor(MemberVisitor memberVisitor)
    {
        ensureArraySize();

        memberVisitors[memberVisitorCount++] = memberVisitor;
    }


    private void ensureArraySize()
    {
        if (memberVisitors == null)
        {
            memberVisitors = new MemberVisitor[ARRAY_SIZE_INCREMENT];
        }
        else if (memberVisitors.length == memberVisitorCount)
        {
            MemberVisitor[] newMemberVisitors =
                new MemberVisitor[memberVisitorCount +
                                         ARRAY_SIZE_INCREMENT];
            System.arraycopy(memberVisitors, 0,
                             newMemberVisitors, 0,
                             memberVisitorCount);
            memberVisitors = newMemberVisitors;
        }
    }


    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        for (int index = 0; index < memberVisitorCount; index++)
        {
            memberVisitors[index].visitProgramField(programClass, programField);
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        for (int index = 0; index < memberVisitorCount; index++)
        {
            memberVisitors[index].visitProgramMethod(programClass, programMethod);
        }
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        for (int index = 0; index < memberVisitorCount; index++)
        {
            memberVisitors[index].visitLibraryField(libraryClass, libraryField);
        }
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        for (int index = 0; index < memberVisitorCount; index++)
        {
            memberVisitors[index].visitLibraryMethod(libraryClass, libraryMethod);
        }
    }
}
