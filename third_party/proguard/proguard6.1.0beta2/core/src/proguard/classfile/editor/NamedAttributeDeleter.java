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
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;


/**
 * This ClassVisitor deletes attributes with a given name in the program
 * classes, fields, methods, or code attributes that it visits.
 *
 * @author Eric Lafortune
 */
public class NamedAttributeDeleter
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             AttributeVisitor
{
    private final String attributeName;


    public NamedAttributeDeleter(String attributeName)
    {
        this.attributeName = attributeName;
    }


    // Implementations for ClassVisitor.

    public void visitLibraryClass(LibraryClass libraryClass) {}


    public void visitProgramClass(ProgramClass programClass)
    {
        new AttributesEditor(programClass, false).deleteAttribute(attributeName);
    }


    // Implementations for MemberVisitor.

    public void visitLibraryMember(LibraryClass libraryClass, LibraryMember libraryMember) {}


    public void visitProgramMember(ProgramClass programClass, ProgramMember programMember)
    {
        new AttributesEditor(programClass, programMember, false).deleteAttribute(attributeName);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        new AttributesEditor((ProgramClass)clazz, (ProgramMember)method, codeAttribute, false).deleteAttribute(attributeName);
    }
}