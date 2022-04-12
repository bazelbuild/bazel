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
import proguard.classfile.attribute.visitor.ParameterInfoVisitor;

/**
 * This ParameterInfoVisitor adds all parameter information that it visits to
 * the given target method parameters attribute.
 */
public class ParameterInfoAdder
implements   ParameterInfoVisitor
{
    private final ConstantAdder             constantAdder;
    private final MethodParametersAttribute targetMethodParametersAttribute;


    /**
     * Creates a new ParameterInfoAdder that will copy parameter information
     * into the given target method parameters attribute.
     */
    public ParameterInfoAdder(ProgramClass              targetClass,
                              MethodParametersAttribute targetMethodParametersAttribute)
    {
        this.constantAdder                   = new ConstantAdder(targetClass);
        this.targetMethodParametersAttribute = targetMethodParametersAttribute;
    }


    // Implementations for ParameterInfoVisitor.

    public void visitParameterInfo(Clazz clazz, Method method, int parameterIndex, ParameterInfo parameterInfo)
    {
        // Create a new parameter.
        int newNameIndex = parameterInfo.u2nameIndex == 0 ? 0 :
            constantAdder.addConstant(clazz, parameterInfo.u2nameIndex);

        ParameterInfo newParameterInfo =
            new ParameterInfo(newNameIndex, parameterInfo.u2accessFlags);

        // Add it to the target.
        targetMethodParametersAttribute.parameters[parameterIndex] = newParameterInfo;
    }
}