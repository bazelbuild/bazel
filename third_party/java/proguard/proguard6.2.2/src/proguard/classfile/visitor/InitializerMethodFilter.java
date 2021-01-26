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
import proguard.classfile.util.ClassUtil;

/**
 * This MemberVisitor delegates its visits to one of two other given
 * MemberVisitor instances, depending on whether the visited method
 * is a static initializer or instance initializer, or not.
 *
 * @author Eric Lafortune
 */
public class InitializerMethodFilter
implements   MemberVisitor
{
    private final MemberVisitor initializerMemberVisitor;
    private final MemberVisitor otherMemberVisitor;


    /**
     * Creates a new InitializerMethodFilter.
     * @param initializerMemberVisitor the member visitor to which visits to
     *                                 initializers will be delegated.
     */
    public InitializerMethodFilter(MemberVisitor initializerMemberVisitor)
    {
        this(initializerMemberVisitor, null);
    }


    /**
     * Creates a new InitializerMethodFilter.
     * @param initializerMemberVisitor the member visitor to which visits to
     *                                 initializers will be delegated.
     * @param otherMemberVisitor       the member visitor to which visits to
     *                                 non-initializer methods will be delegated.
     */
    public InitializerMethodFilter(MemberVisitor initializerMemberVisitor,
                                   MemberVisitor otherMemberVisitor)
    {
        this.initializerMemberVisitor = initializerMemberVisitor;
        this.otherMemberVisitor       = otherMemberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField) {}
    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField) {}


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        MemberVisitor memberVisitor =
            applicableMemberVisitor(programClass, programMethod);

        if (memberVisitor != null)
        {
            memberVisitor.visitProgramMethod(programClass, programMethod);
        }
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        MemberVisitor memberVisitor =
            applicableMemberVisitor(libraryClass, libraryMethod);

        if (memberVisitor != null)
        {
            memberVisitor.visitLibraryMethod(libraryClass, libraryMethod);
        }
    }


    // Small utility methods.

    /**
     * Returns the appropriate member visitor, depending on whether the
     * given method is an initializer or not.
     */
    private MemberVisitor applicableMemberVisitor(Clazz clazz, Member method)
    {
        return ClassUtil.isInitializer(method.getName(clazz)) ?
            initializerMemberVisitor :
            otherMemberVisitor;
    }
}
