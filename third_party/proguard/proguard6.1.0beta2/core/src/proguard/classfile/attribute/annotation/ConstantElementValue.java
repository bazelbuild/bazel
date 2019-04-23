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

import proguard.classfile.Clazz;
import proguard.classfile.attribute.annotation.visitor.ElementValueVisitor;

/**
 * This ElementValue represents a constant element value.
 *
 * @author Eric Lafortune
 */
public class ConstantElementValue extends ElementValue
{
    public final char u1tag;
    public       int  u2constantValueIndex;


    /**
     * Creates an uninitialized ConstantElementValue.
     */
    public ConstantElementValue(char u1tag)
    {
        this.u1tag = u1tag;
    }


    /**
     * Creates an initialized ConstantElementValue.
     */
    public ConstantElementValue(char u1tag,
                                int  u2elementNameIndex,
                                int  u2constantValueIndex)
    {
        super(u2elementNameIndex);

        this.u1tag                = u1tag;
        this.u2constantValueIndex = u2constantValueIndex;
    }


    // Implementations for ElementValue.

    public char getTag()
    {
        return u1tag;
    }

    public void accept(Clazz clazz, Annotation annotation, ElementValueVisitor elementValueVisitor)
    {
        elementValueVisitor.visitConstantElementValue(clazz, annotation, this);
    }
}
