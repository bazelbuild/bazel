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
import proguard.classfile.attribute.visitor.BootstrapMethodInfoVisitor;

/**
 * This BootstrapMethodInfoVisitor adds all bootstrap methods that it visits to
 * the given target bootstrap methods attribute.
 */
public class BootstrapMethodInfoAdder
implements   BootstrapMethodInfoVisitor
{
    private final ConstantAdder                   constantAdder;
    private final BootstrapMethodsAttributeEditor bootstrapMethodsAttributeEditor;

    private int bootstrapMethodIndex;


    /**
     * Creates a new BootstrapMethodInfoAdder that will copy bootstrap methods
     * into the given bootstrap methods attribute.
     */
    public BootstrapMethodInfoAdder(ProgramClass              targetClass,
                                    BootstrapMethodsAttribute targetBootstrapMethodsAttribute)
    {
        this.constantAdder                   = new ConstantAdder(targetClass);
        this.bootstrapMethodsAttributeEditor = new BootstrapMethodsAttributeEditor(targetBootstrapMethodsAttribute);
    }


    /**
     * Returns the index of the most recently added bootstrap method.
     */
    public int getBootstrapMethodIndex()
    {
        return bootstrapMethodIndex;
    }


    // Implementations for BootstrapMethodInfoVisitor.

    public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
    {
        // Copy the method arguments.
        int   methodArgumentCount = bootstrapMethodInfo.u2methodArgumentCount;
        int[] methodArguments     = bootstrapMethodInfo.u2methodArguments;
        int[] newMethodArguments  = new int[methodArgumentCount];

        for (int index = 0; index < methodArgumentCount; index++)
        {
            newMethodArguments[index] =
                constantAdder.addConstant(clazz, methodArguments[index]);
        }

        // Create a new bootstrap method.
        BootstrapMethodInfo newBootstrapMethodInfo =
            new BootstrapMethodInfo(constantAdder.addConstant(clazz, bootstrapMethodInfo.u2methodHandleIndex),
                                    methodArgumentCount,
                                    newMethodArguments);

        // Add it to the target.
        bootstrapMethodIndex =
            bootstrapMethodsAttributeEditor.addBootstrapMethodInfo(newBootstrapMethodInfo);
    }
}