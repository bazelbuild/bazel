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
 * This class can add annotations to a given annotations attribute.
 * Annotations to be added must have been filled out beforehand.
 *
 * @author Eric Lafortune
 */
public class AnnotationsAttributeEditor
{
    private AnnotationsAttribute targetAnnotationsAttribute;


    /**
     * Creates a new AnnotationsAttributeEditor that will edit annotations in
     * the given annotations attribute.
     */
    public AnnotationsAttributeEditor(AnnotationsAttribute targetAnnotationsAttribute)
    {
        this.targetAnnotationsAttribute = targetAnnotationsAttribute;
    }


    /**
     * Adds a given annotation to the annotations attribute.
     */
    public void addAnnotation(Annotation annotation)
    {
        int          annotationsCount = targetAnnotationsAttribute.u2annotationsCount;
        Annotation[] annotations      = targetAnnotationsAttribute.annotations;

        // Make sure there is enough space for the new annotation.
        if (annotations.length <= annotationsCount)
        {
            targetAnnotationsAttribute.annotations = new Annotation[annotationsCount+1];
            System.arraycopy(annotations, 0,
                             targetAnnotationsAttribute.annotations, 0,
                             annotationsCount);
            annotations = targetAnnotationsAttribute.annotations;
        }

        // Add the annotation.
        annotations[targetAnnotationsAttribute.u2annotationsCount++] = annotation;
    }


    /**
     * Deletes a given annotation from the annotations attribute.
     */
    public void deleteAnnotation(Annotation annotation)
    {
        int index = findAnnotationIndex(annotation,
                                        targetAnnotationsAttribute.annotations,
                                        targetAnnotationsAttribute.u2annotationsCount);
        deleteAnnotation(index);
    }


    /**
     * Deletes the annotation at the given idnex from the annotations attribute.
     */
    public void deleteAnnotation(int index)
    {
        ArrayUtil.remove(targetAnnotationsAttribute.annotations,
                         targetAnnotationsAttribute.u2annotationsCount,
                         index);
        targetAnnotationsAttribute.u2annotationsCount--;
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