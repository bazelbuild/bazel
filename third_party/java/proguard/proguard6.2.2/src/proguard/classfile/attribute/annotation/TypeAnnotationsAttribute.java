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
import proguard.classfile.attribute.annotation.visitor.*;

/**
 * This Attribute represents a type annotations attribute.
 *
 * @author Eric Lafortune
 */
public abstract class TypeAnnotationsAttribute extends AnnotationsAttribute
{
    /**
     * Creates an uninitialized TypeAnnotationsAttribute.
     */
    protected TypeAnnotationsAttribute()
    {
    }


    /**
     * Creates an initialized TypeAnnotationsAttribute.
     */
    protected TypeAnnotationsAttribute(int              u2attributeNameIndex,
                                       int              u2annotationsCount,
                                       TypeAnnotation[] annotations)
    {
        super(u2attributeNameIndex, u2annotationsCount, annotations);
    }


    /**
     * Applies the given visitor to all class annotations.
     */
    public void typeAnnotationsAccept(Clazz clazz, TypeAnnotationVisitor typeAnnotationVisitor)
    {
        TypeAnnotation[] annotations = (TypeAnnotation[])this.annotations;

        for (int index = 0; index < u2annotationsCount; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of Annotation.
            typeAnnotationVisitor.visitTypeAnnotation(clazz, annotations[index]);
        }
    }


    /**
     * Applies the given visitor to all field annotations.
     */
    public void typeAnnotationsAccept(Clazz clazz, Field field, TypeAnnotationVisitor typeAnnotationVisitor)
    {
        TypeAnnotation[] annotations = (TypeAnnotation[])this.annotations;

        for (int index = 0; index < u2annotationsCount; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of Annotation.
            typeAnnotationVisitor.visitTypeAnnotation(clazz, field, annotations[index]);
        }
    }


    /**
     * Applies the given visitor to all method annotations.
     */
    public void typeAnnotationsAccept(Clazz clazz, Method method, TypeAnnotationVisitor typeAnnotationVisitor)
    {
        TypeAnnotation[] annotations = (TypeAnnotation[])this.annotations;

        for (int index = 0; index < u2annotationsCount; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of Annotation.
            typeAnnotationVisitor.visitTypeAnnotation(clazz, method, annotations[index]);
        }
    }
}
