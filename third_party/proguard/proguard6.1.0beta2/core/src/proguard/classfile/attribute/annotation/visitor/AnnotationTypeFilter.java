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
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.attribute.annotation.Annotation;
import proguard.util.*;

import java.util.List;

/**
 * This <code>AnnotationVisitor</code> delegates its visits to another given
 * <code>AnnotationVisitor</code>, but only when the visited annotation has
 * a type that matches a given regular expression.
 *
 * @author Eric Lafortune
 */
public class AnnotationTypeFilter
implements   AnnotationVisitor
{
    private final StringMatcher     regularExpressionMatcher;
    private final AnnotationVisitor annotationVisitor;


    /**
     * Creates a new AnnotationTypeFilter.
     * @param regularExpression      the regular expression against which
     *                               annotation type names will be matched.
     * @param annotationVisitor      the annotation visitor to which visits
     *                               will be delegated.
     */
    public AnnotationTypeFilter(String            regularExpression,
                                AnnotationVisitor annotationVisitor)
    {
        this(regularExpression, null, annotationVisitor);
    }


    /**
     * Creates a new AnnotationTypeFilter.
     * @param regularExpression      the regular expression against which
     *                               annotation type names will be matched.
     * @param variableStringMatchers an optional mutable list of
     *                               VariableStringMatcher instances that match
     *                               the wildcards.
     * @param annotationVisitor      the annotation visitor to which visits
     *                               will be delegated.
     */
    public AnnotationTypeFilter(String            regularExpression,
                                List              variableStringMatchers,
                                AnnotationVisitor annotationVisitor)
    {
        this(new ListParser(new ClassNameParser()).parse(regularExpression),
             annotationVisitor);
    }


    /**
     * Creates a new AnnotationTypeFilter.
     * @param regularExpressionMatcher the string matcher against which
     *                                 class names will be matched.
     * @param annotationVisitor        the annotation visitor to which visits
     *                                 will be delegated.
     */
    public AnnotationTypeFilter(StringMatcher     regularExpressionMatcher,
                                AnnotationVisitor annotationVisitor)
    {
        this.regularExpressionMatcher = regularExpressionMatcher;
        this.annotationVisitor        = annotationVisitor;
    }


    // Implementations for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        if (accepted(annotation.getType(clazz)))
        {
            annotationVisitor.visitAnnotation(clazz, annotation);
        }
    }


    public void visitAnnotation(Clazz clazz, Field field, Annotation annotation)
    {
        if (accepted(annotation.getType(clazz)))
        {
            annotationVisitor.visitAnnotation(clazz, field, annotation);
        }
    }


    public void visitAnnotation(Clazz clazz, Method method, Annotation annotation)
    {
        if (accepted(annotation.getType(clazz)))
        {
            annotationVisitor.visitAnnotation(clazz, method, annotation);
        }
    }


    public void visitAnnotation(Clazz clazz, Method method, int parameterIndex, Annotation annotation)
    {
        if (accepted(annotation.getType(clazz)))
        {
            annotationVisitor.visitAnnotation(clazz, method, parameterIndex, annotation);
        }
    }


    public void visitAnnotation(Clazz clazz, Method method, CodeAttribute codeAttribute, Annotation annotation)
    {
        if (accepted(annotation.getType(clazz)))
        {
            annotationVisitor.visitAnnotation(clazz, method, codeAttribute, annotation);
        }
    }


    // Small utility methods.

    private boolean accepted(String name)
    {
        return regularExpressionMatcher.matches(name);
    }
}
