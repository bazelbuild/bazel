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
package proguard.obfuscate;

import proguard.classfile.*;
import proguard.classfile.constant.ClassConstant;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.ConstantPoolEditor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

/**
 * This <code>ClassVisitor</code> renames the class names and class member
 * names of the classes it visits, using names previously determined by the
 * obfuscator.
 *
 * @see ClassObfuscator
 * @see MemberObfuscator
 *
 * @author Eric Lafortune
 */
public class ClassRenamer
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             ConstantVisitor
{
    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Rename this class.
        programClass.thisClassConstantAccept(this);

        // Rename the class members.
        programClass.fieldsAccept(this);
        programClass.methodsAccept(this);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        libraryClass.thisClassName = ClassObfuscator.newClassName(libraryClass);

        // Rename the class members.
        libraryClass.fieldsAccept(this);
        libraryClass.methodsAccept(this);
    }


    // Implementations for MemberVisitor.

    public void visitProgramMember(ProgramClass  programClass,
                                     ProgramMember programMember)
    {
        // Has the class member name changed?
        String name    = programMember.getName(programClass);
        String newName = MemberObfuscator.newMemberName(programMember);
        if (newName != null &&
            !newName.equals(name))
        {
            programMember.u2nameIndex =
                new ConstantPoolEditor(programClass).addUtf8Constant(newName);
        }
    }

    public void visitLibraryMember(LibraryClass  libraryClass,
                                   LibraryMember libraryMember)
    {
        String newName = MemberObfuscator.newMemberName(libraryMember);
        if (newName != null)
        {
            libraryMember.name = newName;
        }
    }


    // Implementations for ConstantVisitor.

    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Update the Class entry if required.
        String newName = ClassObfuscator.newClassName(clazz);
        if (newName != null)
        {
            // Refer to a new Utf8 entry.
            classConstant.u2nameIndex =
                new ConstantPoolEditor((ProgramClass)clazz).addUtf8Constant(newName);
        }
    }
}
