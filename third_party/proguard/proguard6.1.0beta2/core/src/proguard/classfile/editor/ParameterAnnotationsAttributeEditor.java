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

import proguard.classfile.attribute.annotation.*;
import proguard.util.ArrayUtil;

/**
 * This class can add annotations to a given parameter annotations attribute.
 * Annotations to be added must have been filled out beforehand.
 *
 * @author Eric Lafortune
 */
public class ParameterAnnotationsAttributeEditor
{
    private ParameterAnnotationsAttribute targetParameterAnnotationsAttribute;


    /**
     * Creates a new ParameterAnnotationsAttributeEditor that will edit
     * annotations in the given parameter annotations attribute.
     */
    public ParameterAnnotationsAttributeEditor(ParameterAnnotationsAttribute targetParameterAnnotationsAttribute)
    {
        this.targetParameterAnnotationsAttribute = targetParameterAnnotationsAttribute;
    }


    /**
     * Adds a given annotation to the annotations attribute.
     */
    public void addAnnotation(int parameterIndex, Annotation annotation)
    {
        targetParameterAnnotationsAttribute.parameterAnnotations[parameterIndex] =
            ArrayUtil.add(targetParameterAnnotationsAttribute.parameterAnnotations[parameterIndex],
                          targetParameterAnnotationsAttribute.u2parameterAnnotationsCount[parameterIndex]++,
                          annotation);
    }

    /**
     * Deletes a given annotation from the annotations attribute.
     */
    public void deleteAnnotation(int parameterIndex, Annotation annotation)
    {
        int index = findAnnotationIndex(annotation,
                                        targetParameterAnnotationsAttribute.parameterAnnotations[parameterIndex],
                                        targetParameterAnnotationsAttribute.u2parameterAnnotationsCount[parameterIndex]);
        deleteAnnotation(parameterIndex, index);
    }

    /**
     * Deletes the annotation at the given index from the annotations attribute.
     */
    public void deleteAnnotation(int parameterIndex, int annotationIndex)
    {
        ArrayUtil.remove(targetParameterAnnotationsAttribute.parameterAnnotations[parameterIndex],
                         targetParameterAnnotationsAttribute.u2parameterAnnotationsCount[parameterIndex],
                         annotationIndex);
        targetParameterAnnotationsAttribute.u2parameterAnnotationsCount[parameterIndex]--;
    }

    private int findAnnotationIndex(Annotation annotation, Annotation[] annotations, int annotationCount)
    {
        for (int index = 0; index < annotationCount; index++)
        {
            if (annotation == annotations[index])
            {
                return index;
            }

        }
        return -1;
    }
}