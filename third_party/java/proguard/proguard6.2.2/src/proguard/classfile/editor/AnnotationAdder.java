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
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.annotation.visitor.AnnotationVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This AnnotationVisitor adds all annotations that it visits to the given
 * target annotation element value, target annotation attribute, or target
 * parameter annotation attribute.
 *
 * @author Eric Lafortune
 */
public class AnnotationAdder
extends      SimplifiedVisitor
implements   AnnotationVisitor
{
    private static final ElementValue[] EMPTY_ELEMENT_VALUES = new ElementValue[0];


    private final ProgramClass                        targetClass;
    private final AnnotationElementValue              targetAnnotationElementValue;
    private final AnnotationsAttributeEditor          annotationsAttributeEditor;
    private final ParameterAnnotationsAttributeEditor parameterAnnotationsAttributeEditor;

    private final ConstantAdder constantAdder;


    /**
     * Creates a new AnnotationAdder that will copy annotations into the given
     * target annotation element value.
     */
    public AnnotationAdder(ProgramClass           targetClass,
                           AnnotationElementValue targetAnnotationElementValue)
    {
        this.targetClass                         = targetClass;
        this.targetAnnotationElementValue        = targetAnnotationElementValue;
        this.annotationsAttributeEditor          = null;
        this.parameterAnnotationsAttributeEditor = null;

        constantAdder = new ConstantAdder(targetClass);
    }


    /**
     * Creates a new AnnotationAdder that will copy annotations into the given
     * target annotations attribute.
     */
    public AnnotationAdder(ProgramClass         targetClass,
                           AnnotationsAttribute targetAnnotationsAttribute)
    {
        this.targetClass                         = targetClass;
        this.targetAnnotationElementValue        = null;
        this.annotationsAttributeEditor          = new AnnotationsAttributeEditor(targetAnnotationsAttribute);
        this.parameterAnnotationsAttributeEditor = null;

        constantAdder = new ConstantAdder(targetClass);
    }


    /**
     * Creates a new AnnotationAdder that will copy annotations into the given
     * target parameter annotations attribute.
     */
    public AnnotationAdder(ProgramClass                  targetClass,
                           ParameterAnnotationsAttribute targetParameterAnnotationsAttribute)
    {
        this.targetClass                         = targetClass;
        this.targetAnnotationElementValue        = null;
        this.annotationsAttributeEditor          = null;
        this.parameterAnnotationsAttributeEditor = new ParameterAnnotationsAttributeEditor(targetParameterAnnotationsAttribute);

        constantAdder = new ConstantAdder(targetClass);
    }


    // Implementations for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        Annotation newAnnotation =
            new Annotation(constantAdder.addConstant(clazz, annotation.u2typeIndex),
                           0,
                           annotation.u2elementValuesCount > 0 ?
                               new ElementValue[annotation.u2elementValuesCount] :
                               EMPTY_ELEMENT_VALUES);

        // TODO: Clone array.
        newAnnotation.referencedClasses = annotation.referencedClasses;

        // Add the element values.
        annotation.elementValuesAccept(clazz,
                                       new ElementValueAdder(targetClass,
                                                             newAnnotation,
                                                             false));

        // What's the target?
        if (targetAnnotationElementValue != null)
        {
            // Simply set the completed annotation.
            targetAnnotationElementValue.annotationValue = newAnnotation;
        }
        else
        {
            // Add the completed annotation.
            annotationsAttributeEditor.addAnnotation(newAnnotation);
        }
    }


    public void visitAnnotation(Clazz clazz, Method method, int parameterIndex, Annotation annotation)
    {
        Annotation newAnnotation =
            new Annotation(constantAdder.addConstant(clazz, annotation.u2typeIndex),
                           0,
                           annotation.u2elementValuesCount > 0 ?
                               new ElementValue[annotation.u2elementValuesCount] :
                               EMPTY_ELEMENT_VALUES);

        // TODO: Clone array.
        newAnnotation.referencedClasses = annotation.referencedClasses;

        // Add the element values.
        annotation.elementValuesAccept(clazz,
                                       new ElementValueAdder(targetClass,
                                                             newAnnotation,
                                                             false));

        // Add the completed annotation.
        parameterAnnotationsAttributeEditor.addAnnotation(parameterIndex, newAnnotation);
    }
}