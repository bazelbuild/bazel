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
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.editor.ConstantPoolEditor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This ClassVisitor changes the name stored in the source file attributes
 * and source dir attributes of the classes that it visits, if the
 * attributes are present.
 *
 * @author Eric Lafortune
 */
public class SourceFileRenamer
extends      SimplifiedVisitor
implements   ClassVisitor,
             AttributeVisitor
{
    private final String newSourceFileAttribute;


    /**
     * Creates a new SourceFileRenamer.
     * @param newSourceFileAttribute the new string to be put in the source file
     *                               attributes.
     */
    public SourceFileRenamer(String newSourceFileAttribute)
    {
        this.newSourceFileAttribute = newSourceFileAttribute;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Only visit the class attributes.
        programClass.attributesAccept(this);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitSourceFileAttribute(Clazz clazz, SourceFileAttribute sourceFileAttribute)
    {
        // Fix the source file attribute.
        sourceFileAttribute.u2sourceFileIndex =
            new ConstantPoolEditor((ProgramClass)clazz).addUtf8Constant(newSourceFileAttribute);
    }


    public void visitSourceDirAttribute(Clazz clazz, SourceDirAttribute sourceDirAttribute)
    {
        // Fix the source file attribute.
        sourceDirAttribute.u2sourceDirIndex =
            new ConstantPoolEditor((ProgramClass)clazz).addUtf8Constant(newSourceFileAttribute);
    }
}
