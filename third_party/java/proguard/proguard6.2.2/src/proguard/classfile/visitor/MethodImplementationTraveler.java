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
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This <code>MemberVisitor</code> lets a given <code>MemberVisitor</code>
 * travel to all concrete and abstract implementations of the visited methods
 * in their class hierarchies.
 *
 * @author Eric Lafortune
 */
public class MethodImplementationTraveler
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final boolean       visitThisMethod;
    private final boolean       visitSuperMethods;
    private final boolean       visitInterfaceMethods;
    private final boolean       visitOverridingMethods;
    private final MemberVisitor memberVisitor;


    /**
     * Creates a new MethodImplementationTraveler.
     * @param visitThisMethod        specifies whether to visit the originally
     *                               visited methods.
     * @param visitSuperMethods      specifies whether to visit the method in
     *                               the super classes.
     * @param visitInterfaceMethods  specifies whether to visit the method in
     *                               the interface classes.
     * @param visitOverridingMethods specifies whether to visit the method in
     *                               the subclasses.
     * @param memberVisitor          the <code>MemberVisitor</code> to which
     *                               visits will be delegated.
     */
    public MethodImplementationTraveler(boolean       visitThisMethod,
                                        boolean       visitSuperMethods,
                                        boolean       visitInterfaceMethods,
                                        boolean       visitOverridingMethods,
                                        MemberVisitor memberVisitor)
    {
        this.visitThisMethod        = visitThisMethod;
        this.visitSuperMethods      = visitSuperMethods;
        this.visitInterfaceMethods  = visitInterfaceMethods;
        this.visitOverridingMethods = visitOverridingMethods;
        this.memberVisitor          = memberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if (visitThisMethod)
        {
            programMethod.accept(programClass, memberVisitor);
        }

        if (!isSpecial(programClass, programMethod))
        {
            programClass.hierarchyAccept(false,
                                         visitSuperMethods,
                                         visitInterfaceMethods,
                                         visitOverridingMethods,
                                         new NamedMethodVisitor(programMethod.getName(programClass),
                                                                programMethod.getDescriptor(programClass),
                                         new MemberAccessFilter(0,
                                                                ClassConstants.ACC_PRIVATE |
                                                                ClassConstants.ACC_STATIC,
                                         memberVisitor)));
        }
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        if (visitThisMethod)
        {
            libraryMethod.accept(libraryClass, memberVisitor);
        }

        if (!isSpecial(libraryClass, libraryMethod))
        {
            libraryClass.hierarchyAccept(false,
                                         visitSuperMethods,
                                         visitInterfaceMethods,
                                         visitOverridingMethods,
                                         new NamedMethodVisitor(libraryMethod.getName(libraryClass),
                                                                libraryMethod.getDescriptor(libraryClass),
                                         new MemberAccessFilter(0,
                                                                ClassConstants.ACC_PRIVATE |
                                                                ClassConstants.ACC_STATIC,
                                         memberVisitor)));
        }
    }


    // Small utility methods.

    private boolean isSpecial(Clazz clazz, Method method)
    {
        return (method.getAccessFlags() &
                (ClassConstants.ACC_PRIVATE |
                 ClassConstants.ACC_STATIC)) != 0 ||
               method.getName(clazz).equals(ClassConstants.METHOD_NAME_INIT);
    }
}
