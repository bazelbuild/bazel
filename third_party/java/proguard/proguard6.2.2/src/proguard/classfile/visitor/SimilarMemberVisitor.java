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
 * This <code>MemberVisitor</code> lets a given <code>MemberVisitor</code>
 * visit all members that have the same name and type as the visited methods
 * in the class hierarchy of the members' classes or of a given target class.
 *
 * @author Eric Lafortune
 */
public class SimilarMemberVisitor
implements   MemberVisitor
{
    private final Clazz         targetClass;
    private final boolean       visitThisMember;
    private final boolean       visitSuperMembers;
    private final boolean       visitInterfaceMembers;
    private final boolean       visitOverridingMembers;
    private final MemberVisitor memberVisitor;


    /**
     * Creates a new SimilarMemberVisitor.
     * @param visitThisMember        specifies whether to visit the class
     *                               members in the members' classes themselves.
     * @param visitSuperMembers      specifies whether to visit the class
     *                               members in the super classes of the
     *                               members' classes.
     * @param visitInterfaceMembers  specifies whether to visit the class
     *                               members in the interface classes of the
     *                               members' classes.
     * @param visitOverridingMembers specifies whether to visit the class
     *                               members in the subclasses of the members'
     *                               classes.
     * @param memberVisitor          the <code>MemberVisitor</code> to which
     *                               visits will be delegated.
     */
    public SimilarMemberVisitor(boolean       visitThisMember,
                                boolean       visitSuperMembers,
                                boolean       visitInterfaceMembers,
                                boolean       visitOverridingMembers,
                                MemberVisitor memberVisitor)
    {
        this(null,
             visitThisMember,
             visitSuperMembers,
             visitInterfaceMembers,
             visitOverridingMembers,
             memberVisitor);
    }


    /**
     * Creates a new SimilarMemberVisitor.
     * @param targetClass            the class in whose hierarchy to look for
     *                               the visited class members.
     * @param visitThisMember        specifies whether to visit the class
     *                               members in the target class itself.
     * @param visitSuperMembers      specifies whether to visit the class
     *                               members in the super classes of the target
     *                               class.
     * @param visitInterfaceMembers  specifies whether to visit the class
     *                               members in the interface classes of the
     *                               target class.
     * @param visitOverridingMembers specifies whether to visit the class
     *                               members in the subclasses of the target
     *                               class.
     * @param memberVisitor          the <code>MemberVisitor</code> to which
     *                               visits will be delegated.
     */
    public SimilarMemberVisitor(Clazz         targetClass,
                                boolean       visitThisMember,
                                boolean       visitSuperMembers,
                                boolean       visitInterfaceMembers,
                                boolean       visitOverridingMembers,
                                MemberVisitor memberVisitor)
    {
        this.targetClass            = targetClass;
        this.visitThisMember        = visitThisMember;
        this.visitSuperMembers      = visitSuperMembers;
        this.visitInterfaceMembers  = visitInterfaceMembers;
        this.visitOverridingMembers = visitOverridingMembers;
        this.memberVisitor          = memberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        Clazz targetClass = targetClass(programClass);

        targetClass.hierarchyAccept(visitThisMember,
                                    visitSuperMembers,
                                    visitInterfaceMembers,
                                    visitOverridingMembers,
                                    new NamedFieldVisitor(programField.getName(programClass),
                                                          programField.getDescriptor(programClass),
                                                          memberVisitor));
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        Clazz targetClass = targetClass(libraryClass);

        targetClass.hierarchyAccept(visitThisMember,
                                    visitSuperMembers,
                                    visitInterfaceMembers,
                                    visitOverridingMembers,
                                    new NamedFieldVisitor(libraryField.getName(libraryClass),
                                                          libraryField.getDescriptor(libraryClass),
                                                          memberVisitor));
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        Clazz targetClass = targetClass(programClass);

        targetClass.hierarchyAccept(visitThisMember,
                                    visitSuperMembers,
                                    visitInterfaceMembers,
                                    visitOverridingMembers,
                                    new NamedMethodVisitor(programMethod.getName(programClass),
                                                           programMethod.getDescriptor(programClass),
                                                           memberVisitor));
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        Clazz targetClass = targetClass(libraryClass);

        targetClass.hierarchyAccept(visitThisMember,
                                    visitSuperMembers,
                                    visitInterfaceMembers,
                                    visitOverridingMembers,
                                    new NamedMethodVisitor(libraryMethod.getName(libraryClass),
                                                           libraryMethod.getDescriptor(libraryClass),
                                                           memberVisitor));
    }


    /**
     * Returns the target class, or the given class if the target class is
     * null.
     */
    private Clazz targetClass(Clazz clazz)
    {
        return targetClass != null ? targetClass : clazz;
    }
}