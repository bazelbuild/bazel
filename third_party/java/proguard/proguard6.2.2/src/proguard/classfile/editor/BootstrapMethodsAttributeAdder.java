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
 * the given target class, creating a bootstrap methods attribute if necessary.
 */
public class BootstrapMethodsAttributeAdder
implements   BootstrapMethodInfoVisitor
{
    private final ProgramClass             targetClass;
    private final ConstantPoolEditor       constantPoolEditor;
    private       BootstrapMethodInfoAdder bootstrapMethodInfoAdder;


    /**
     * Creates a new BootstrapMethodsAttributeAdder that will copy bootstrap
     * methods into the given target class/
     */
    public BootstrapMethodsAttributeAdder(ProgramClass targetClass)
    {
        this.targetClass        = targetClass;
        this.constantPoolEditor = new ConstantPoolEditor(targetClass);
    }


    /**
     * Returns the index of the most recently added bootstrap method.
     */
    public int getBootstrapMethodIndex()
    {
        return bootstrapMethodInfoAdder.getBootstrapMethodIndex();
    }


    // Implementations for BootstrapMethodInfoVisitor.

    public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
    {
        // Make sure we have a bootstrap methods attribute adder.
        if (bootstrapMethodInfoAdder == null)
        {
            // Make sure we have a target bootstrap methods attribute.
            AttributesEditor attributesEditor =
                new AttributesEditor(targetClass, false);

            BootstrapMethodsAttribute targetBootstrapMethodsAttribute =
                (BootstrapMethodsAttribute)attributesEditor.findAttribute(ClassConstants.ATTR_BootstrapMethods);

            if (targetBootstrapMethodsAttribute == null)
            {
                targetBootstrapMethodsAttribute =
                    new BootstrapMethodsAttribute(constantPoolEditor.addUtf8Constant(ClassConstants.ATTR_BootstrapMethods),
                                                  0,
                                                  new BootstrapMethodInfo[0]);

                attributesEditor.addAttribute(targetBootstrapMethodsAttribute);
            }

            // Create a bootstrap method adder for it.
            bootstrapMethodInfoAdder = new BootstrapMethodInfoAdder(targetClass,
                                                                    targetBootstrapMethodsAttribute);
        }

        // Delegate to the bootstrap method adder.
        bootstrapMethodInfoAdder.visitBootstrapMethodInfo(clazz, bootstrapMethodInfo);
    }
}