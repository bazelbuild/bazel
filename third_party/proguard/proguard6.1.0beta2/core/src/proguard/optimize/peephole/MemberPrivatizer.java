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
package proguard.optimize.peephole;

import proguard.classfile.*;
import proguard.classfile.editor.MethodInvocationFixer;
import proguard.classfile.util.*;
import proguard.classfile.visitor.MemberVisitor;
import proguard.optimize.info.NonPrivateMemberMarker;

/**
 * This MemberVisitor makes all class members that it visits private, unless
 * they have been marked by a NonPrivateMemberMarker. The invocations of
 * privatized methods still have to be fixed.
 *
 * @see NonPrivateMemberMarker
 * @see MethodInvocationFixer
 * @author Eric Lafortune
 */
public class MemberPrivatizer
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final MemberVisitor extraMemberVisitor;


    /**
     * Creates a new MemberPrivatizer.
     */
    public MemberPrivatizer()
    {
        this(null);
    }


    /**
     * Creates a new MemberPrivatizer.
     * @param extraMemberVisitor an optional extra visitor for all privatized
     *                           class members.
     */
    public MemberPrivatizer(MemberVisitor extraMemberVisitor)
    {
        this.extraMemberVisitor = extraMemberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        // Is the field unmarked?
        if (NonPrivateMemberMarker.canBeMadePrivate(programField))
        {
            // Make the field private.
            programField.u2accessFlags =
                AccessUtil.replaceAccessFlags(programField.u2accessFlags,
                                              ClassConstants.ACC_PRIVATE);

            // Visit the field, if required.
            if (extraMemberVisitor != null)
            {
                extraMemberVisitor.visitProgramField(programClass, programField);
            }
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        // Is the method unmarked?
        if (NonPrivateMemberMarker.canBeMadePrivate(programMethod))
        {
            // Make the method private and no longer final.
            programMethod.u2accessFlags =
                AccessUtil.replaceAccessFlags(programMethod.u2accessFlags,
                                              ClassConstants.ACC_PRIVATE);

            // Visit the method, if required.
            if (extraMemberVisitor != null)
            {
                extraMemberVisitor.visitProgramMethod(programClass, programMethod);
            }
        }
    }
}
