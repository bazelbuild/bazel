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
import proguard.classfile.attribute.Attribute;
import proguard.classfile.attribute.annotation.visitor.AnnotationVisitor;

/**
 * This Attribute represents a parameter annotations attribute.
 *
 * @author Eric Lafortune
 */
public abstract class ParameterAnnotationsAttribute extends Attribute
{
    // Note that the java compilers of JDK 1.5+ and of Eclipse count the
    // number of parameters of constructors of non-static inner classes and
    // of enum classes, based on the Signature attribute (if any), which
    // lacks the first one or two synthetic parameters. Unresolved issues:
    // https://bugs.java.com/bugdatabase/view_bug.do?bug_id=8024694
    // https://bugs.java.com/bugdatabase/view_bug.do?bug_id=8024694
    public int            u1parametersCount;
    public int[]          u2parameterAnnotationsCount;
    public Annotation[][] parameterAnnotations;


    /**
     * Creates an uninitialized ParameterAnnotationsAttribute.
     */
    protected ParameterAnnotationsAttribute()
    {
    }


    /**
     * Creates an initialized ParameterAnnotationsAttribute.
     */
    protected ParameterAnnotationsAttribute(int            u2attributeNameIndex,
                                            int            u1parametersCount,
                                            int[]          u2parameterAnnotationsCount,
                                            Annotation[][] parameterAnnotations)
    {
        super(u2attributeNameIndex);

        this.u1parametersCount           = u1parametersCount;
        this.u2parameterAnnotationsCount = u2parameterAnnotationsCount;
        this.parameterAnnotations        = parameterAnnotations;
    }


    /**
     * Applies the given visitor to all annotations.
     */
    public void annotationsAccept(Clazz clazz, Method method, AnnotationVisitor annotationVisitor)
    {
        // Loop over all parameters.
        for (int parameterIndex = 0; parameterIndex < u1parametersCount; parameterIndex++)
        {
            int          annotationsCount = u2parameterAnnotationsCount[parameterIndex];
            Annotation[] annotations      = parameterAnnotations[parameterIndex];

            // Loop over all parameter annotations.
            for (int index = 0; index < annotationsCount; index++)
            {
                // We don't need double dispatching here, since there is only one
                // type of Annotation.
                annotationVisitor.visitAnnotation(clazz, method, parameterIndex, annotations[index]);
            }
        }
    }
}
