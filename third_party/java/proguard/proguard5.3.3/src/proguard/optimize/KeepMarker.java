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
package proguard.optimize;

import proguard.classfile.*;
import proguard.classfile.util.MethodLinker;
import proguard.classfile.visitor.*;
import proguard.optimize.info.NoSideEffectMethodMarker;


/**
 * This <code>ClassVisitor</code> and <code>MemberVisitor</code>
 * marks classes and class members it visits. The marked elements
 * will remain unchanged as necessary in the optimization step.
 *
 * @see NoSideEffectMethodMarker
 * @author Eric Lafortune
 */
public class KeepMarker
implements   ClassVisitor,
             MemberVisitor
{
    // A visitor info flag to indicate the visitor accepter is being kept.
    private static final Object KEPT = new Object();


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        markAsKept(programClass);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        markAsKept(libraryClass);
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        markAsKept(programField);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        markAsKept(MethodLinker.lastMember(programMethod));
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        markAsKept(libraryField);
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        markAsKept(MethodLinker.lastMember(libraryMethod));
    }


    // Small utility methods.

    private static void markAsKept(VisitorAccepter visitorAccepter)
    {
        visitorAccepter.setVisitorInfo(KEPT);
    }


    public static boolean isKept(VisitorAccepter visitorAccepter)
    {
        // We're also checking for the constant in NoSideEffectMethodMarker,
        // to keep things simple.
        Object visitorInfo =
            MethodLinker.lastVisitorAccepter(visitorAccepter).getVisitorInfo();

        return visitorInfo == KEPT ||
               visitorInfo == NoSideEffectMethodMarker.KEPT_BUT_NO_SIDE_EFFECTS;
    }
}
