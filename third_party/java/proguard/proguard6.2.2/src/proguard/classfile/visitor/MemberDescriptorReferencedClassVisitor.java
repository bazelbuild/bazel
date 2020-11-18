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
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This MemberVisitor lets a given ClassVisitor visit all the classes
 * referenced by the descriptors of the class members that it visits.
 *
 * @author Eric Lafortune
 */
public class MemberDescriptorReferencedClassVisitor
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final ClassVisitor classVisitor;


    public MemberDescriptorReferencedClassVisitor(ClassVisitor classVisitor)
    {
        this.classVisitor = classVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramMember(ProgramClass programClass, ProgramMember programMember)
    {
        // Let the visitor visit the classes referenced in the descriptor string.
        programMember.referencedClassesAccept(classVisitor);
    }


    public void visitLibraryMember(LibraryClass programClass, LibraryMember libraryMember)
    {
        // Let the visitor visit the classes referenced in the descriptor string.
        libraryMember.referencedClassesAccept(classVisitor);
    }
}
