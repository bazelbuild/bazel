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
package proguard.classfile.attribute.annotation;

import proguard.classfile.*;
import proguard.classfile.attribute.annotation.visitor.ElementValueVisitor;

/**
 * This ElementValue represents an array element value.
 *
 * @author Eric Lafortune
 */
public class ArrayElementValue extends ElementValue
{
    public int            u2elementValuesCount;
    public ElementValue[] elementValues;


    /**
     * Creates an uninitialized ArrayElementValue.
     */
    public ArrayElementValue()
    {
    }


    /**
     * Creates an initialized ArrayElementValue.
     */
    public ArrayElementValue(int            u2elementNameIndex,
                             int            u2elementValuesCount,
                             ElementValue[] elementValues)
    {
        super(u2elementNameIndex);

        this.u2elementValuesCount = u2elementValuesCount;
        this.elementValues        = elementValues;
    }


    // Implementations for ElementValue.

    public char getTag()
    {
        return ClassConstants.ELEMENT_VALUE_ARRAY;
    }

    public void accept(Clazz clazz, Annotation annotation, ElementValueVisitor elementValueVisitor)
    {
        elementValueVisitor.visitArrayElementValue(clazz, annotation, this);
    }


    /**
     * Applies the given visitor to all nested element values.
     */
    public void elementValuesAccept(Clazz clazz, Annotation annotation, ElementValueVisitor elementValueVisitor)
    {
        for (int index = 0; index < u2elementValuesCount; index++)
        {
            elementValues[index].accept(clazz, annotation, elementValueVisitor);
        }
    }
}
