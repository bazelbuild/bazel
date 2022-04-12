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
import proguard.classfile.visitor.MemberVisitor;
import proguard.optimize.info.*;

/**
 * This <code>MemberVisitor</code> delegates its visits to another given
 * <code>MemberVisitor</code>, but only when the visited member has editable
 * optimization info.
 *
 * @see FieldOptimizationInfo
 * @see ProgramFieldOptimizationInfo
 * @see MethodOptimizationInfo
 * @see ProgramMethodOptimizationInfo
 * @author Eric Lafortune
 */
public class OptimizationInfoMemberFilter
implements   MemberVisitor
{
    private final MemberVisitor memberVisitor;
    private final MemberVisitor otherMemberVisitor;


    /**
     * Creates a new OptimizationInfoMemberFilter.
     * @param memberVisitor the <code>MemberVisitor</code> to which visits will
     *                      be delegated.
     */
    public OptimizationInfoMemberFilter(MemberVisitor memberVisitor)
    {
        this(memberVisitor, null);
    }


    /**
     * Creates a new OptimizationInfoMemberFilter.
     * @param memberVisitor         the <code>MemberVisitor</code> to which visits will
     *                              be delegated if the member has editable optimization
     *                              info.
     * @param otherMemberVisitor    the <code>MemberVisitor</code> to which visits will
     *                              be delegated if the member does not have editable
     *                              optimization info.
     */
    public OptimizationInfoMemberFilter(MemberVisitor memberVisitor,
                                        MemberVisitor otherMemberVisitor)
    {
        this.memberVisitor      = memberVisitor;
        this.otherMemberVisitor = otherMemberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField) {}
    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod) {}

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        MemberVisitor visitor =
            FieldOptimizationInfo.getFieldOptimizationInfo(programField) instanceof ProgramFieldOptimizationInfo ?
                memberVisitor : otherMemberVisitor;

        if (visitor != null)
        {
            visitor.visitProgramField(programClass, programField);
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        MemberVisitor visitor =
            MethodOptimizationInfo.getMethodOptimizationInfo(programMethod) instanceof ProgramMethodOptimizationInfo ?
                memberVisitor : otherMemberVisitor;

        if (visitor != null)
        {
            visitor.visitProgramMethod(programClass, programMethod);
        }
    }
}
