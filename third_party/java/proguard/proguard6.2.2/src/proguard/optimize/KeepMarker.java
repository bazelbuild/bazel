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
package proguard.optimize;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;
import proguard.optimize.info.*;

/**
 * This <code>ClassVisitor</code>, <code>MemberVisitor</code> and
 * <code>AttributeVisitor</code> marks classes, class members and
 * code attributes it visits. The marked elements will remain
 * unchanged as necessary in the optimization step.
 *
 * @see NoSideEffectMethodMarker
 * @author Eric Lafortune
 */
public class KeepMarker
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             AttributeVisitor
{
    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        ClassOptimizationInfo.setClassOptimizationInfo(programClass);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        ClassOptimizationInfo.setClassOptimizationInfo(libraryClass);
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        FieldOptimizationInfo.setFieldOptimizationInfo(programClass, programField);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        MethodOptimizationInfo.setMethodOptimizationInfo(programClass, programMethod);
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        FieldOptimizationInfo.setFieldOptimizationInfo(libraryClass, libraryField);
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        MethodOptimizationInfo.setMethodOptimizationInfo(libraryClass, libraryMethod);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        CodeAttributeOptimizationInfo.setCodeAttributeOptimizationInfo(codeAttribute);
    }


    // Small utility methods.

    public static boolean isKept(Clazz clazz)
    {
        ClassOptimizationInfo info =
            ClassOptimizationInfo.getClassOptimizationInfo(clazz);

        return info != null &&
               info.isKept();
    }

    public static boolean isKept(Field field)
    {
        FieldOptimizationInfo info =
            FieldOptimizationInfo.getFieldOptimizationInfo(field);

        return info != null &&
               info.isKept();
    }

    public static boolean isKept(Method method)
    {
        MethodOptimizationInfo info =
            MethodOptimizationInfo.getMethodOptimizationInfo(method);

        return info != null &&
               info.isKept();
    }

    public static boolean isKept(CodeAttribute codeAttribute)
    {
        CodeAttributeOptimizationInfo info =
            CodeAttributeOptimizationInfo.getCodeAttributeOptimizationInfo(codeAttribute);

        return info != null &&
               info.isKept();
    }

}
