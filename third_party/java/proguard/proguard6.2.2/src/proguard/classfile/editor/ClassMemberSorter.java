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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.visitor.ClassVisitor;

import java.util.*;

/**
 * This ClassVisitor sorts the class members of the classes that it visits.
 * The sorting order is based on the access flags, the names, and the
 * descriptors.
 *
 * @author Eric Lafortune
 */
public class ClassMemberSorter implements ClassVisitor, Comparator
{
    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Sort the fields.
        Arrays.sort(programClass.fields, 0, programClass.u2fieldsCount, this);

        // Sort the methods.
        Arrays.sort(programClass.methods, 0, programClass.u2methodsCount, this);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
    }


    // Implementations for Comparator.

    public int compare(Object object1, Object object2)
    {
        ProgramMember member1 = (ProgramMember)object1;
        ProgramMember member2 = (ProgramMember)object2;

        return member1.u2accessFlags     < member2.u2accessFlags     ? -1 :
               member1.u2accessFlags     > member2.u2accessFlags     ?  1 :
               member1.u2nameIndex       < member2.u2nameIndex       ? -1 :
               member1.u2nameIndex       > member2.u2nameIndex       ?  1 :
               member1.u2descriptorIndex < member2.u2descriptorIndex ? -1 :
               member1.u2descriptorIndex > member2.u2descriptorIndex ?  1 :
                                                                        0;
    }
}
