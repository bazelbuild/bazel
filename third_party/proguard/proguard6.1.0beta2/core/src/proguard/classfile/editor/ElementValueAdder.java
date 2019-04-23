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
import proguard.classfile.attribute.annotation.visitor.ElementValueVisitor;

/**
 * This AnnotationVisitor adds all element values that it visits to the given
 * target annotation default attribute, annotation, or element value.
 *
 * @author Eric Lafortune
 */
public class ElementValueAdder
implements   ElementValueVisitor
{
    private static final ElementValue[] EMPTY_ELEMENT_VALUES = new ElementValue[0];


    private final ProgramClass               targetClass;
    private final AnnotationDefaultAttribute targetAnnotationDefaultAttribute;

    private final ConstantAdder       constantAdder;
    private final ElementValuesEditor elementValuesEditor;


    /**
     * Creates a new ElementValueAdder that will copy element values into the
     * given target annotation default attribute value.
     */
    public ElementValueAdder(ProgramClass               targetClass,
                             AnnotationDefaultAttribute targetAnnotationDefaultAttribute,
                             boolean                    replaceElementValues)
    {
        this.targetClass                      = targetClass;
        this.targetAnnotationDefaultAttribute = targetAnnotationDefaultAttribute;

        constantAdder       = new ConstantAdder(targetClass);
        elementValuesEditor = null;
    }


    /**
     * Creates a new ElementValueAdder that will copy element values into the
     * given target annotation.
     */
    public ElementValueAdder(ProgramClass targetClass,
                             Annotation   targetAnnotation,
                             boolean      replaceElementValues)
    {
        this.targetClass                      = targetClass;
        this.targetAnnotationDefaultAttribute = null;

        constantAdder       = new ConstantAdder(targetClass);
        elementValuesEditor = new ElementValuesEditor(targetClass,
                                                      targetAnnotation,
                                                      replaceElementValues);
    }


    /**
     * Creates a new ElementValueAdder that will copy element values into the
     * given target element value.
     */
    public ElementValueAdder(ProgramClass      targetClass,
                             ArrayElementValue targetArrayElementValue,
                             boolean           replaceElementValues)
    {
        this.targetClass                      = targetClass;
        this.targetAnnotationDefaultAttribute = null;

        constantAdder       = new ConstantAdder(targetClass);
        elementValuesEditor = new ElementValuesEditor(targetClass,
                                                      targetArrayElementValue,
                                                      replaceElementValues);
    }


    // Implementations for ElementValueVisitor.

    public void visitConstantElementValue(Clazz clazz, Annotation annotation, ConstantElementValue constantElementValue)
    {
        // Create a copy of the element value.
        ConstantElementValue newConstantElementValue =
            new ConstantElementValue(constantElementValue.u1tag,
                                     constantElementValue.u2elementNameIndex == 0 ? 0 :
                                     constantAdder.addConstant(clazz, constantElementValue.u2elementNameIndex),
                                     constantAdder.addConstant(clazz, constantElementValue.u2constantValueIndex));

        newConstantElementValue.referencedClass  = constantElementValue.referencedClass;
        newConstantElementValue.referencedMethod = constantElementValue.referencedMethod;

        // Add it to the target.
        addElementValue(newConstantElementValue);
    }


    public void visitEnumConstantElementValue(Clazz clazz, Annotation annotation, EnumConstantElementValue enumConstantElementValue)
    {
        // Create a copy of the element value.
        EnumConstantElementValue newEnumConstantElementValue =
            new EnumConstantElementValue(enumConstantElementValue.u2elementNameIndex == 0 ? 0 :
                                         constantAdder.addConstant(clazz, enumConstantElementValue.u2elementNameIndex),
                                         constantAdder.addConstant(clazz, enumConstantElementValue.u2typeNameIndex),
                                         constantAdder.addConstant(clazz, enumConstantElementValue.u2constantNameIndex));

        newEnumConstantElementValue.referencedClass  = enumConstantElementValue.referencedClass;
        newEnumConstantElementValue.referencedMethod = enumConstantElementValue.referencedMethod;

        // TODO: Clone array.
        newEnumConstantElementValue.referencedClasses = enumConstantElementValue.referencedClasses;

        // Add it to the target.
        addElementValue(newEnumConstantElementValue);
    }


    public void visitClassElementValue(Clazz clazz, Annotation annotation, ClassElementValue classElementValue)
    {
        // Create a copy of the element value.
        ClassElementValue newClassElementValue =
            new ClassElementValue(classElementValue.u2elementNameIndex == 0 ? 0 :
                                  constantAdder.addConstant(clazz, classElementValue.u2elementNameIndex),
                                  constantAdder.addConstant(clazz, classElementValue.u2classInfoIndex));

        newClassElementValue.referencedClass  = classElementValue.referencedClass;
        newClassElementValue.referencedMethod = classElementValue.referencedMethod;

        // TODO: Clone array.
        newClassElementValue.referencedClasses = classElementValue.referencedClasses;

        // Add it to the target.
        addElementValue(newClassElementValue);
    }


    public void visitAnnotationElementValue(Clazz clazz, Annotation annotation, AnnotationElementValue annotationElementValue)
    {
        // Create a copy of the element value.
        AnnotationElementValue newAnnotationElementValue =
            new AnnotationElementValue(annotationElementValue.u2elementNameIndex == 0 ? 0 :
                                       constantAdder.addConstant(clazz, annotationElementValue.u2elementNameIndex),
                                       new Annotation());

        newAnnotationElementValue.referencedClass  = annotationElementValue.referencedClass;
        newAnnotationElementValue.referencedMethod = annotationElementValue.referencedMethod;

        annotationElementValue.annotationAccept(clazz,
                                                new AnnotationAdder(targetClass,
                                                                    newAnnotationElementValue));

        // Add it to the target.
        addElementValue(newAnnotationElementValue);
    }


    public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
    {
        // Create a copy of the element value.
        ArrayElementValue newArrayElementValue =
            new ArrayElementValue(arrayElementValue.u2elementNameIndex == 0 ? 0 :
                                  constantAdder.addConstant(clazz, arrayElementValue.u2elementNameIndex),
                                  0,
                                  arrayElementValue.u2elementValuesCount > 0 ?
                                      new ElementValue[arrayElementValue.u2elementValuesCount] :
                                      EMPTY_ELEMENT_VALUES);

        newArrayElementValue.referencedClass  = arrayElementValue.referencedClass;
        newArrayElementValue.referencedMethod = arrayElementValue.referencedMethod;

        arrayElementValue.elementValuesAccept(clazz,
                                              annotation,
                                              new ElementValueAdder(targetClass,
                                                                    newArrayElementValue,
                                                                    false));

        // Add it to the target.
        addElementValue(newArrayElementValue);
    }


    // Small utility methods.

    private void addElementValue(ElementValue newElementValue)
    {
        // What's the target?
        if (targetAnnotationDefaultAttribute != null)
        {
            // Simply set the completed element value.
            targetAnnotationDefaultAttribute.defaultValue = newElementValue;
        }
        else
        {
            // Add it to the target.
            elementValuesEditor.addElementValue(newElementValue);
        }
    }
}