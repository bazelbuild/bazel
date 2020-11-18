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
package proguard.optimize.gson;

import proguard.classfile.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;

/**
 * This visitor searches the classes that it visits for fields that can be
 * involved in Json (de)serialization and passes them on to the given member
 * visitor.
 *
 * For convenience, the classes that are visited are also passed on to the
 * given class visitor.
 *
 * @author Lars Vandenbergh
 */
public class OptimizedJsonFieldVisitor
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor
{
    private final ClassVisitor  classVisitor;
    private final MemberVisitor memberVisitor;


    /**
     * Creates a new OptimizedJsonFieldVisitor.
     *
     * @param classVisitor  the visitor to which (de)serialized classes are
     *                      delegated to.
     * @param memberVisitor the visitor to which (de)serialized fields
     *                      are delegated to.
     */
    public OptimizedJsonFieldVisitor(ClassVisitor  classVisitor,
                                     MemberVisitor memberVisitor)
    {
        this.classVisitor  = classVisitor;
        this.memberVisitor = memberVisitor;
    }

    // Implementations for ClassVisitor.

    @Override
    public void visitAnyClass(Clazz clazz) {}


    @Override
    public void visitProgramClass(ProgramClass programClass)
    {
        programClass.accept(classVisitor);
        programClass.accept(new ClassAccessFilter(0, ClassConstants.ACC_ENUM,
                            new AllFieldVisitor(
                            new MemberAccessFilter(0, ClassConstants.ACC_TRANSIENT,
                            this))));

        // For enums, only visit the enum constant fields.
        programClass.accept(new ClassAccessFilter(ClassConstants.ACC_ENUM, 0,
                            new AllFieldVisitor(
                            new MemberAccessFilter(0, ClassConstants.ACC_TRANSIENT,
                            new MemberDescriptorFilter(ClassUtil.internalTypeFromClassName(programClass.getName()),
                            this)))));
    }


    // Implementations for MemberVisitor.

    @Override
    public void visitAnyMember(Clazz clazz, Member member) {}


    @Override
    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        programField.accept(programClass, memberVisitor);
    }
}
