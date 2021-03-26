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
package proguard.classfile.attribute.annotation.target.visitor;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.annotation.target.*;

/**
 * This interface specifies the methods for a visitor of <code>TargetInfo</code>
 * objects.
 *
 * @author Eric Lafortune
 */
public interface TargetInfoVisitor
{
    public void visitTypeParameterTargetInfo(     Clazz clazz,                                             TypeAnnotation typeAnnotation, TypeParameterTargetInfo      typeParameterTargetInfo);
    public void visitTypeParameterTargetInfo(     Clazz clazz, Method method,                              TypeAnnotation typeAnnotation, TypeParameterTargetInfo      typeParameterTargetInfo);
    public void visitSuperTypeTargetInfo(         Clazz clazz,                                             TypeAnnotation typeAnnotation, SuperTypeTargetInfo          superTypeTargetInfo);
    public void visitTypeParameterBoundTargetInfo(Clazz clazz,                                             TypeAnnotation typeAnnotation, TypeParameterBoundTargetInfo typeParameterBoundTargetInfo);
    public void visitTypeParameterBoundTargetInfo(Clazz clazz, Field field,                                TypeAnnotation typeAnnotation, TypeParameterBoundTargetInfo typeParameterBoundTargetInfo);
    public void visitTypeParameterBoundTargetInfo(Clazz clazz, Method method,                              TypeAnnotation typeAnnotation, TypeParameterBoundTargetInfo typeParameterBoundTargetInfo);
    public void visitEmptyTargetInfo(             Clazz clazz, Field field,                                TypeAnnotation typeAnnotation, EmptyTargetInfo              emptyTargetInfo);
    public void visitEmptyTargetInfo(             Clazz clazz, Method method,                              TypeAnnotation typeAnnotation, EmptyTargetInfo              emptyTargetInfo);
    public void visitFormalParameterTargetInfo(   Clazz clazz, Method method,                              TypeAnnotation typeAnnotation, FormalParameterTargetInfo    formalParameterTargetInfo);
    public void visitThrowsTargetInfo(            Clazz clazz, Method method,                              TypeAnnotation typeAnnotation, ThrowsTargetInfo             throwsTargetInfo);
    public void visitLocalVariableTargetInfo(     Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, LocalVariableTargetInfo      localVariableTargetInfo);
    public void visitCatchTargetInfo(             Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, CatchTargetInfo              catchTargetInfo);
    public void visitOffsetTargetInfo(            Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, OffsetTargetInfo             offsetTargetInfo);
    public void visitTypeArgumentTargetInfo(      Clazz clazz, Method method, CodeAttribute codeAttribute, TypeAnnotation typeAnnotation, TypeArgumentTargetInfo       typeArgumentTargetInfo);
}
