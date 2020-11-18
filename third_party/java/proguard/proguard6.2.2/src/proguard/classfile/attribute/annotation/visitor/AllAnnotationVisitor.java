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
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This AttributeVisitor lets a given AnnotationVisitor visit all Annotation
 * objects of the attributes it visits.
 *
 * @author Eric Lafortune
 */
public class AllAnnotationVisitor
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    private final AnnotationVisitor annotationVisitor;


    /**
     * Creates a new AllAnnotationVisitor.
     * @param annotationVisitor the AnnotationVisitor to which visits will be
     *                          delegated.
     */
    public AllAnnotationVisitor(AnnotationVisitor annotationVisitor)
    {
        this.annotationVisitor = annotationVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeVisibleAnnotationsAttribute.annotationsAccept(clazz, annotationVisitor);
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeVisibleAnnotationsAttribute.annotationsAccept(clazz, field, annotationVisitor);
    }


    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeVisibleAnnotationsAttribute.annotationsAccept(clazz, method, annotationVisitor);
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeInvisibleAnnotationsAttribute.annotationsAccept(clazz, annotationVisitor);
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeInvisibleAnnotationsAttribute.annotationsAccept(clazz, field, annotationVisitor);
    }


    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeInvisibleAnnotationsAttribute.annotationsAccept(clazz, method, annotationVisitor);
    }


    public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
    {
        // Visit the annotations.
        parameterAnnotationsAttribute.annotationsAccept(clazz, method, annotationVisitor);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeVisibleTypeAnnotationsAttribute.annotationsAccept(clazz, annotationVisitor);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeVisibleTypeAnnotationsAttribute.annotationsAccept(clazz, field, annotationVisitor);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeVisibleTypeAnnotationsAttribute.annotationsAccept(clazz, method, annotationVisitor);
    }


    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeVisibleTypeAnnotationsAttribute.annotationsAccept(clazz, method, codeAttribute, annotationVisitor);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeInvisibleTypeAnnotationsAttribute.annotationsAccept(clazz, annotationVisitor);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Field field, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeInvisibleTypeAnnotationsAttribute.annotationsAccept(clazz, field, annotationVisitor);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeInvisibleTypeAnnotationsAttribute.annotationsAccept(clazz, method, annotationVisitor);
    }


    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        // Visit the annotations.
        runtimeInvisibleTypeAnnotationsAttribute.annotationsAccept(clazz, method, codeAttribute, annotationVisitor);
    }
}
