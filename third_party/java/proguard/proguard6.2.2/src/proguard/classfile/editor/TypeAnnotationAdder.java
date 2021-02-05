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
import proguard.classfile.attribute.annotation.target.TargetInfo;
import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This TypeAnnotationVisitor adds all type annotations that it visits to the given
 * target annotation element value, target annotation attribute, or target
 * parameter annotation attribute.
 *
 * @author Eric Lafortune
 */
public class TypeAnnotationAdder
extends      SimplifiedVisitor
implements   TypeAnnotationVisitor
{
    private static final ElementValue[] EMPTY_ELEMENT_VALUES = new ElementValue[0];


    private final ProgramClass               targetClass;
    private final AnnotationElementValue     targetAnnotationElementValue;
    private final AnnotationsAttributeEditor annotationsAttributeEditor;

    private final ConstantAdder constantAdder;


    /**
     * Creates a new TypeAnnotationAdder that will copy annotations into the given
     * target annotation element value.
     */
    public TypeAnnotationAdder(ProgramClass           targetClass,
                               AnnotationElementValue targetAnnotationElementValue)
    {
        this.targetClass                         = targetClass;
        this.targetAnnotationElementValue        = targetAnnotationElementValue;
        this.annotationsAttributeEditor          = null;

        constantAdder = new ConstantAdder(targetClass);
    }


    /**
     * Creates a new TypeAnnotationAdder that will copy annotations into the given
     * target annotations attribute.
     */
    public TypeAnnotationAdder(ProgramClass         targetClass,
                               AnnotationsAttribute targetAnnotationsAttribute)
    {
        this.targetClass                         = targetClass;
        this.targetAnnotationElementValue        = null;
        this.annotationsAttributeEditor          = new AnnotationsAttributeEditor(targetAnnotationsAttribute);

        constantAdder = new ConstantAdder(targetClass);
    }


    // Implementations for AnnotationVisitor.

    public void visitTypeAnnotation(Clazz clazz, TypeAnnotation typeAnnotation)
    {
        TypePathInfo[] typePath    = typeAnnotation.typePath;
        TypePathInfo[] newTypePath = new TypePathInfo[typePath.length];

        TypeAnnotation newTypeAnnotation =
            new TypeAnnotation(constantAdder.addConstant(clazz, typeAnnotation.u2typeIndex),
                               0,
                               typeAnnotation.u2elementValuesCount > 0 ?
                                   new ElementValue[typeAnnotation.u2elementValuesCount] :
                                   EMPTY_ELEMENT_VALUES,
                               null,
                               newTypePath);

        // TODO: Clone array.
        newTypeAnnotation.referencedClasses = typeAnnotation.referencedClasses;

        // Add the element values.
        typeAnnotation.elementValuesAccept(clazz,
                                           new ElementValueAdder(targetClass,
                                                                 newTypeAnnotation,
                                                                 false));

        // Set the target info.
        typeAnnotation.targetInfo.accept(clazz,
                                         typeAnnotation,
                                         new TargetInfoCopier(targetClass, newTypeAnnotation));

        // Copy the type path.
        for (int index = 0; index < typePath.length; index++)
        {
            TypePathInfo typePathInfo    = typePath[index];
            TypePathInfo newTypePathInfo = new TypePathInfo(typePathInfo.u1typePathKind,
                                                            typePathInfo.u1typeArgumentIndex);

            newTypePath[index] = newTypePathInfo;
        }

        // What's the target?
        if (targetAnnotationElementValue != null)
        {
            // Simply set the completed annotation.
            targetAnnotationElementValue.annotationValue = newTypeAnnotation;
        }
        else
        {
            // Add the completed annotation.
            annotationsAttributeEditor.addAnnotation(newTypeAnnotation);
        }
    }
}