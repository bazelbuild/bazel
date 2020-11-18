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

import proguard.classfile.ProgramClass;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This ClassVisitor sorts the various elements of the classes that it visits:
 * interfaces, constants, fields, methods, and attributes.
 *
 * @author Eric Lafortune
 */
public class ClassElementSorter
extends      SimplifiedVisitor
implements   ClassVisitor
{
    private final ClassVisitor interfaceSorter    = new InterfaceSorter();
    private final ClassVisitor constantPoolSorter = new ConstantPoolSorter();
//  private ClassVisitor classMemberSorter  = new ClassMemberSorter();
    private final ClassVisitor attributeSorter    = new AttributeSorter();


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        programClass.accept(constantPoolSorter);
        programClass.accept(interfaceSorter);
//      programClass.accept(classMemberSorter);
        programClass.accept(attributeSorter);
    }
}
