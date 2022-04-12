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
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.annotation.visitor.AnnotationVisitor;

/**
 * This Attribute represents an annotations attribute.
 *
 * @author Eric Lafortune
 */
public abstract class AnnotationsAttribute extends Attribute
{
    public int          u2annotationsCount;
    public Annotation[] annotations;


    /**
     * Creates an uninitialized AnnotationsAttribute.
     */
    protected AnnotationsAttribute()
    {
    }


    /**
     * Creates an initialized AnnotationsAttribute.
     */
    protected AnnotationsAttribute(int          u2attributeNameIndex,
                                   int          u2annotationsCount,
                                   Annotation[] annotations)
    {
        super(u2attributeNameIndex);

        this.u2annotationsCount = u2annotationsCount;
        this.annotations        = annotations;
    }


    /**
     * Applies the given visitor to all class annotations.
     */
    public void annotationsAccept(Clazz clazz, AnnotationVisitor annotationVisitor)
    {
        for (int index = 0; index < u2annotationsCount; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of Annotation.
            annotationVisitor.visitAnnotation(clazz, annotations[index]);
        }
    }


    /**
     * Applies the given visitor to all field annotations.
     */
    public void annotationsAccept(Clazz clazz, Field field, AnnotationVisitor annotationVisitor)
    {
        for (int index = 0; index < u2annotationsCount; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of Annotation.
            annotationVisitor.visitAnnotation(clazz, field, annotations[index]);
        }
    }


    /**
     * Applies the given visitor to all method annotations.
     */
    public void annotationsAccept(Clazz clazz, Method method, AnnotationVisitor annotationVisitor)
    {
        for (int index = 0; index < u2annotationsCount; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of Annotation.
            annotationVisitor.visitAnnotation(clazz, method, annotations[index]);
        }
    }


    /**
     * Applies the given visitor to all code attribute annotations.
     */
    public void annotationsAccept(Clazz clazz, Method method, CodeAttribute codeAttribute, AnnotationVisitor annotationVisitor)
    {
        for (int index = 0; index < u2annotationsCount; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of Annotation.
            annotationVisitor.visitAnnotation(clazz, method, codeAttribute, annotations[index]);
        }
    }
}
