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
package proguard.classfile.attribute;

import proguard.classfile.*;
import proguard.classfile.attribute.visitor.*;

/**
 * This Attribute represents a method parameters attribute.
 *
 * @author Eric Lafortune
 */
public class MethodParametersAttribute extends Attribute
{
    public int             u1parametersCount;
    public ParameterInfo[] parameters;


    /**
     * Creates an uninitialized MethodParametersAttribute.
     */
    public MethodParametersAttribute()
    {
    }


    /**
     * Creates an initialized MethodParametersAttribute.
     */
    public MethodParametersAttribute(int             u2attributeNameIndex,
                                     int             u1parametersCount,
                                     ParameterInfo[] parameters)
    {
        super(u2attributeNameIndex);

        this.u1parametersCount = u1parametersCount;
        this.parameters        = parameters;
    }


    // Implementations for Attribute.

    public void accept(Clazz clazz, Method method, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitMethodParametersAttribute(clazz, method, this);
    }


    /**
     * Applies the given visitor to all parameters.
     */
    public void parametersAccept(Clazz clazz, Method method, ParameterInfoVisitor parameterInfoVisitor)
    {
        // Loop over all parameters.
        for (int index = 0; index < u1parametersCount; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of ParameterInfo.
            parameterInfoVisitor.visitParameterInfo(clazz, method, index, parameters[index]);
        }
    }
}
