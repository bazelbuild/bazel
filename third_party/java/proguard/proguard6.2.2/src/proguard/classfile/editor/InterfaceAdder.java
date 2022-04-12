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
import proguard.classfile.constant.ClassConstant;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This ConstantVisitor adds all interfaces that it visits to the given
 * target class.
 *
 * @author Eric Lafortune
 */
public class InterfaceAdder
extends      SimplifiedVisitor
implements   ConstantVisitor
{
    private final ConstantAdder    constantAdder;
    private final InterfacesEditor interfacesEditor;


    /**
     * Creates a new InterfaceAdder that will add interfaces to the given
     * target class.
     */
    public InterfaceAdder(ProgramClass targetClass)
    {
        constantAdder    = new ConstantAdder(targetClass);
        interfacesEditor = new InterfacesEditor(targetClass);
    }


    // Implementations for ConstantVisitor.

    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        interfacesEditor.addInterface(constantAdder.addConstant(clazz, classConstant));
    }
}