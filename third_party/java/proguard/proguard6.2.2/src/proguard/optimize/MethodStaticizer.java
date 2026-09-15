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
package proguard.optimize;

import proguard.classfile.*;
import proguard.classfile.editor.MethodInvocationFixer;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.MemberVisitor;
import proguard.optimize.info.ParameterUsageMarker;
import proguard.optimize.peephole.VariableShrinker;

/**
 * This MemberVisitor makes all methods that it visits static, if their 'this'
 * parameters are unused.
 *
 * @see ParameterUsageMarker
 * @see MethodInvocationFixer
 * @see VariableShrinker
 * @author Eric Lafortune
 */
public class MethodStaticizer
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final MemberVisitor extraStaticMemberVisitor;


    /**
     * Creates a new MethodStaticizer.
     */
    public MethodStaticizer()
    {
        this(null);
    }


    /**
     * Creates a new MethodStaticizer with an extra visitor.
     * @param extraStaticMemberVisitor an optional extra visitor for all
     *                                 methods that have been made static.
     */
    public MethodStaticizer(MemberVisitor extraStaticMemberVisitor)
    {
        this.extraStaticMemberVisitor = extraStaticMemberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        // Is the 'this' parameter being used?
        if (!ParameterUsageMarker.isParameterUsed(programMethod, 0))
        {
            // Make the method static.
            programMethod.u2accessFlags =
                (programMethod.getAccessFlags() & ~ClassConstants.ACC_FINAL) |
                ClassConstants.ACC_STATIC;

            // Visit the method, if required.
            if (extraStaticMemberVisitor != null)
            {
                extraStaticMemberVisitor.visitProgramMethod(programClass, programMethod);
            }
        }
    }
}
