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
package proguard.classfile.attribute.annotation.visitor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.annotation.*;
import proguard.util.ArrayUtil;

/**
 * This AnnotationVisitor delegates all visits to each AnnotationVisitor
 * in a given list.
 *
 * @author Thomas Neidhart
 */
public class MultiAnnotationVisitor implements AnnotationVisitor
{
    private AnnotationVisitor[] annotationVisitors;
    private int                 annotationVisitorCount;


    public MultiAnnotationVisitor()
    {
        this.annotationVisitors = new AnnotationVisitor[16];
    }


    public MultiAnnotationVisitor(AnnotationVisitor... annotationVisitors)
    {
        this.annotationVisitors     = annotationVisitors;
        this.annotationVisitorCount = annotationVisitors.length;
    }


    public void addAnnotationVisitor(AnnotationVisitor annotationVisitor)
    {
        annotationVisitors =
            ArrayUtil.add(annotationVisitors,
                          annotationVisitorCount++,
                          annotationVisitor);
    }


    // Implementations for AnnotationVisitor.


    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        for (int index = 0; index < annotationVisitorCount; index++)
        {
            annotationVisitors[index].visitAnnotation(clazz, annotation);
        }
    }


    public void visitAnnotation(Clazz clazz, Field field, Annotation annotation)
    {
        for (int index = 0; index < annotationVisitorCount; index++)
        {
            annotationVisitors[index].visitAnnotation(clazz, field, annotation);
        }
    }


    public void visitAnnotation(Clazz clazz, Method method, Annotation annotation)
    {
        for (int index = 0; index < annotationVisitorCount; index++)
        {
            annotationVisitors[index].visitAnnotation(clazz, method, annotation);
        }
    }


    public void visitAnnotation(Clazz clazz, Method method, int parameterIndex, Annotation annotation)
    {
        for (int index = 0; index < annotationVisitorCount; index++)
        {
            annotationVisitors[index].visitAnnotation(clazz, method, parameterIndex, annotation);
        }
    }


    public void visitAnnotation(Clazz clazz, Method method, CodeAttribute codeAttribute, Annotation annotation)
    {
        for (int index = 0; index < annotationVisitorCount; index++)
        {
            annotationVisitors[index].visitAnnotation(clazz, method, codeAttribute, annotation);
        }
    }
}
