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
import proguard.classfile.util.*;
import proguard.classfile.visitor.MemberVisitor;
import proguard.optimize.KeepMarker;

/**
 * This <code>MemberVisitor</code> makes the program methods that it visits
 * final, if possible.
 *
 * @author Eric Lafortune
 */
public class MethodFinalizer
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final MemberVisitor extraMemberVisitor;

    private final MemberFinder memberFinder = new MemberFinder();


    /**
     * Creates a new ClassFinalizer.
     */
    public MethodFinalizer()
    {
        this(null);
    }


    /**
     * Creates a new ClassFinalizer.
     * @param extraMemberVisitor an optional extra visitor for all finalized
     *                           methods.
     */
    public MethodFinalizer(MemberVisitor extraMemberVisitor)
    {
        this.extraMemberVisitor = extraMemberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        String name = programMethod.getName(programClass);

        // If the method is not already private/static/final/abstract,
        // and it is not a constructor,
        // and its class is final,
        //     or it is not being kept and it is not overridden,
        // then make it final.
        if ((programMethod.u2accessFlags & (ClassConstants.ACC_PRIVATE |
                                            ClassConstants.ACC_STATIC  |
                                            ClassConstants.ACC_FINAL   |
                                            ClassConstants.ACC_ABSTRACT)) == 0 &&
            !name.equals(ClassConstants.METHOD_NAME_INIT)                      &&
            ((programClass.u2accessFlags & ClassConstants.ACC_FINAL) != 0 ||
             (!KeepMarker.isKept(programMethod) &&
              (programClass.subClasses == null ||
               !memberFinder.isOverriden(programClass, programMethod)))))
        {
            programMethod.u2accessFlags |= ClassConstants.ACC_FINAL;

            // Visit the method, if required.
            if (extraMemberVisitor != null)
            {
                extraMemberVisitor.visitProgramMethod(programClass, programMethod);
            }
        }
    }
}