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
import proguard.classfile.attribute.annotation.*;

/**
 * This interface specifies the methods for a visitor of <code>ElementValue</code>
 * objects.
 *
 * @author Eric Lafortune
 */
public interface ElementValueVisitor
{
    public void visitConstantElementValue(    Clazz clazz,                Annotation annotation, ConstantElementValue     constantElementValue);
    public void visitEnumConstantElementValue(Clazz clazz,                Annotation annotation, EnumConstantElementValue enumConstantElementValue);
    public void visitClassElementValue(       Clazz clazz,                Annotation annotation, ClassElementValue        classElementValue);
    public void visitAnnotationElementValue(  Clazz clazz,                Annotation annotation, AnnotationElementValue   annotationElementValue);
    public void visitArrayElementValue(       Clazz clazz,                Annotation annotation, ArrayElementValue        arrayElementValue);

//    public void visitConstantElementValue(    Clazz clazz, Field  field,  Annotation annotation, ConstantElementValue     constantElementValue);
//    public void visitEnumConstantElementValue(Clazz clazz, Field  field,  Annotation annotation, EnumConstantElementValue enumConstantElementValue);
//    public void visitClassElementValue(       Clazz clazz, Field  field,  Annotation annotation, ClassElementValue        classElementValue);
//    public void visitAnnotationElementValue(  Clazz clazz, Field  field,  Annotation annotation, AnnotationElementValue   annotationElementValue);
//    public void visitArrayElementValue(       Clazz clazz, Field  field,  Annotation annotation, ArrayElementValue        arrayElementValue);
//
//    public void visitConstantElementValue(    Clazz clazz, Method method, Annotation annotation, ConstantElementValue     constantElementValue);
//    public void visitEnumConstantElementValue(Clazz clazz, Method method, Annotation annotation, EnumConstantElementValue enumConstantElementValue);
//    public void visitClassElementValue(       Clazz clazz, Method method, Annotation annotation, ClassElementValue        classElementValue);
//    public void visitAnnotationElementValue(  Clazz clazz, Method method, Annotation annotation, AnnotationElementValue   annotationElementValue);
//    public void visitArrayElementValue(       Clazz clazz, Method method, Annotation annotation, ArrayElementValue        arrayElementValue);
}
