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
 * This visitor clears the specified access flags of the
 * program classes and class members that its visits.
 *
 * @see ClassConstants
 *
 * @author Eric Lafortune
 */
public class MemberAccessFlagCleaner
implements   ClassVisitor,
             MemberVisitor
{
    private final int accessFlags;


    /**
     * Creates a new MemberAccessFlagCleaner.
     * @param accessFlags the member access flags to be cleared.
     */
    public MemberAccessFlagCleaner(int accessFlags)
    {
        this.accessFlags = accessFlags;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        programClass.u2accessFlags &= ~accessFlags;
    }

    public void visitLibraryClass(LibraryClass libraryClass)
    {
        libraryClass.u2accessFlags &= ~accessFlags;
    }


    // Implementations for MemberVisitor.

    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField) {}
    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod) {}


    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        programField.u2accessFlags &= ~accessFlags;
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        programMethod.u2accessFlags &= ~accessFlags;
    }
}
