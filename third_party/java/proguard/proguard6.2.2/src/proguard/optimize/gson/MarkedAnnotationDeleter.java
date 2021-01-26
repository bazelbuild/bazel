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
package proguard.optimize.gson;

import proguard.classfile.*;
import proguard.classfile.attribute.Attribute;
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.util.SimplifiedVisitor;


/**
 * This AttributeVisitor deletes annotations with the given object as
 * visitorInfo on the attributes that it visits.
 * If deleting an annotation results in the corresponding annotation attribute
 * to be empty, that attribute will be deleted as well.
 *
 * @author Rob Coekaerts
 */
class      MarkedAnnotationDeleter
extends    SimplifiedVisitor
implements AttributeVisitor
{
    // A visitor info flag to indicate the annotation can be deleted.
    private final Object mark;


    /**
     * Creates a new MarkedAnnotationDeleter.
     *
     * @param mark the visitor info used to recognize annotations that
     *             need to be deleted.
     */
    public MarkedAnnotationDeleter(Object mark)
    {
        this.mark = mark;
    }


    // Implementations for AttributeVisitor.


    @Override
    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    @Override
    public void visitRuntimeVisibleAnnotationsAttribute(Clazz clazz,
                                                        Member member,
                                                        RuntimeVisibleAnnotationsAttribute runtimeVisibleAnnotationsAttribute)
    {
        cleanAnnotationsAttribute(clazz,
                                  member,
                                  runtimeVisibleAnnotationsAttribute,
                                  ClassConstants.ATTR_RuntimeVisibleAnnotations);
    }


    @Override
    public void visitRuntimeInvisibleAnnotationsAttribute(Clazz clazz,
                                                          Member member,
                                                          RuntimeInvisibleAnnotationsAttribute runtimeInvisibleAnnotationsAttribute)
    {
        cleanAnnotationsAttribute(clazz,
                                  member,
                                  runtimeInvisibleAnnotationsAttribute,
                                  ClassConstants.ATTR_RuntimeInvisibleAnnotations);
    }


    @Override
    public void visitRuntimeVisibleParameterAnnotationsAttribute(Clazz clazz,
                                                                 Method method,
                                                                 RuntimeVisibleParameterAnnotationsAttribute runtimeVisibleParameterAnnotationsAttribute)
    {
        cleanParameterAnnotationsAttribute(clazz,
                                           method,
                                           runtimeVisibleParameterAnnotationsAttribute,
                                           ClassConstants.ATTR_RuntimeVisibleParameterAnnotations);
    }


    @Override
    public void visitRuntimeInvisibleParameterAnnotationsAttribute(Clazz clazz,
                                                                   Method method,
                                                                   RuntimeInvisibleParameterAnnotationsAttribute runtimeInvisibleParameterAnnotationsAttribute)
    {
        cleanParameterAnnotationsAttribute(clazz,
                                           method,
                                           runtimeInvisibleParameterAnnotationsAttribute,
                                           ClassConstants.ATTR_RuntimeInvisibleParameterAnnotations);
    }


    @Override
    public void visitRuntimeVisibleTypeAnnotationsAttribute(Clazz clazz,
                                                            Member member,
                                                            RuntimeVisibleTypeAnnotationsAttribute runtimeVisibleTypeAnnotationsAttribute)
    {
        cleanAnnotationsAttribute(clazz,
                                  member,
                                  runtimeVisibleTypeAnnotationsAttribute,
                                  ClassConstants.ATTR_RuntimeVisibleTypeAnnotations);
    }


    @Override
    public void visitRuntimeInvisibleTypeAnnotationsAttribute(Clazz clazz,
                                                              Member member,
                                                              RuntimeInvisibleTypeAnnotationsAttribute runtimeInvisibleTypeAnnotationsAttribute)
    {
        cleanAnnotationsAttribute(clazz,
                                  member,
                                  runtimeInvisibleTypeAnnotationsAttribute,
                                  ClassConstants.ATTR_RuntimeInvisibleTypeAnnotations);
    }


    // Utility methods


    private void cleanAnnotationsAttribute(Clazz                clazz,
                                           Member               member,
                                           AnnotationsAttribute attribute,
                                           String               attributeName)
    {
        // Delete marked annotations.
        AnnotationsAttributeEditor annotationsAttributeEditor = new AnnotationsAttributeEditor(attribute);
        Annotation[]               annotations                = attribute.annotations;
        for (int index = 0; index < attribute.u2annotationsCount; index++)
        {
            Annotation annotation = annotations[index];
            if (annotation.getVisitorInfo() == mark)
            {
                annotationsAttributeEditor.deleteAnnotation(index);
            }
        }

        // Delete attribute if no annotations are left.
        if (attribute.u2annotationsCount == 0)
        {
            AttributesEditor attributesEditor = new AttributesEditor((ProgramClass) clazz,
                                                                     (ProgramMember)member,
                                                                     false);
            attributesEditor.deleteAttribute(attributeName);
        }
    }


    private void cleanParameterAnnotationsAttribute(Clazz                         clazz,
                                                    Member                        member,
                                                    ParameterAnnotationsAttribute attribute,
                                                    String                        attributeName)
    {
        // Delete marked annotations.
        ParameterAnnotationsAttributeEditor annotationsAttributeEditor =
            new ParameterAnnotationsAttributeEditor(attribute);
        boolean allEmpty = true;
        for (int parameterIndex = 0; parameterIndex < attribute.u1parametersCount; parameterIndex++)
        {
            int          annotationsCount = attribute.u2parameterAnnotationsCount[parameterIndex];
            Annotation[] annotations      = attribute.parameterAnnotations[parameterIndex];
            for (int annotationIndex = 0; annotationIndex < annotationsCount; annotationIndex++)
            {
                Annotation annotation = annotations[annotationIndex];
                if (annotation.getVisitorInfo() == mark)
                {
                    annotationsAttributeEditor.deleteAnnotation(parameterIndex, annotationIndex);
                }
            }
            if (attribute.u2parameterAnnotationsCount[parameterIndex] != 0)
            {
                allEmpty = false;
            }
        }

        // Delete attribute if all parameters have no annotations left.
        if (allEmpty)
        {
            AttributesEditor attributesEditor = new AttributesEditor((ProgramClass) clazz,
                                                                     (ProgramMember)member,
                                                                     false);
            attributesEditor.deleteAttribute(attributeName);
        }
    }
}
