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

import proguard.classfile.Clazz;
import proguard.classfile.attribute.annotation.Annotation;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;


/**
 * This AnnotationVisitor delegates all visits to a given ClassVisitor.
 * The latter visits the class of each visited annotation, although
 * never twice in a row.
 *
 * @author Eric Lafortune
 */
public class AnnotationToAnnotatedClassVisitor
extends      SimplifiedVisitor
implements   AnnotationVisitor
{
    private final ClassVisitor classVisitor;

    private Clazz lastVisitedClass;


    public AnnotationToAnnotatedClassVisitor(ClassVisitor classVisitor)
    {
        this.classVisitor = classVisitor;
    }


    // Implementations for AnnotationVisitor.

    public void visitAnnotation(Clazz clazz, Annotation annotation)
    {
        if (!clazz.equals(lastVisitedClass))
        {
            clazz.accept(classVisitor);

            lastVisitedClass = clazz;
        }
    }
}
