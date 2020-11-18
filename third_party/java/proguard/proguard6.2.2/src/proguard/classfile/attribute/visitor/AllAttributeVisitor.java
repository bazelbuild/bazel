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
package proguard.classfile.attribute.visitor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

/**
 * This ClassVisitor, MemberVisitor, and AttributeVisitor lets a given
 * AttributeVisitor visit all Attribute objects of the program classes,
 * program class members, or code attributes, respectively, that it visits.
 *
 * @author Eric Lafortune
 */
public class AllAttributeVisitor
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             AttributeVisitor
{
    private final boolean          deep;
    private final AttributeVisitor attributeVisitor;


    /**
     * Creates a new shallow AllAttributeVisitor.
     * @param attributeVisitor the AttributeVisitor to which visits will be
     *                         delegated.
     */
    public AllAttributeVisitor(AttributeVisitor attributeVisitor)
    {
        this(false, attributeVisitor);
    }


    /**
     * Creates a new optionally deep AllAttributeVisitor.
     * @param deep             specifies whether the attributes contained
     *                         further down the class structure should be
     *                         visited too.
     * @param attributeVisitor the AttributeVisitor to which visits will be
     *                         delegated.
     */
    public AllAttributeVisitor(boolean          deep,
                               AttributeVisitor attributeVisitor)
    {
        this.deep             = deep;
        this.attributeVisitor = attributeVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        programClass.attributesAccept(attributeVisitor);

        // Visit the attributes further down the class structure, if required.
        if (deep)
        {
            programClass.fieldsAccept(this);
            programClass.methodsAccept(this);
            programClass.attributesAccept(this);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass) {}


    // Implementations for MemberVisitor.

    public void visitProgramMember(ProgramClass programClass, ProgramMember programMember)
    {
        programMember.attributesAccept(programClass, attributeVisitor);

        // Visit the attributes further down the member structure, if required.
        if (deep)
        {
            programMember.attributesAccept(programClass, this);
        }
    }


    public void visitLibraryMember(LibraryClass programClass, LibraryMember programMember) {}


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        codeAttribute.attributesAccept(clazz, method, attributeVisitor);
    }
}
