/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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
package proguard.optimize.info;

import proguard.classfile.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

/**
 * This ClassVisitor marks all class members that can not be made private in the
 * classes that it visits, and in the classes to which they refer.
 *
 * @author Eric Lafortune
 */
public class NonPrivateMemberMarker
extends      SimplifiedVisitor
implements   ClassVisitor,
             ConstantVisitor,
             MemberVisitor
{
    private final MethodImplementationFilter filteredMethodMarker = new MethodImplementationFilter(this);


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Mark all referenced class members in different classes.
        programClass.constantPoolEntriesAccept(this);

        // Explicitly mark the <clinit> method.
        programClass.methodAccept(ClassConstants.METHOD_NAME_CLINIT,
                                  ClassConstants.METHOD_TYPE_CLINIT,
                                  this);

        // Explicitly mark the parameterless <init> method.
        programClass.methodAccept(ClassConstants.METHOD_NAME_INIT,
                                  ClassConstants.METHOD_TYPE_INIT,
                                  this);

        // Mark all methods that may have implementations.
        programClass.methodsAccept(filteredMethodMarker);
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Go over all methods.
        libraryClass.methodsAccept(this);
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        // The referenced class member, if any, can never be made private,
        // even if it's in the same class.
        stringConstant.referencedMemberAccept(this);
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        Clazz referencedClass = refConstant.referencedClass;

        // Is it referring to a class member in another class?
        // The class member might be in another class, or
        // it may be referenced through another class.
        if (referencedClass != null &&
            !referencedClass.equals(clazz) ||
            !refConstant.getClassName(clazz).equals(clazz.getName()))
        {
            // The referenced class member can never be made private.
            refConstant.referencedMemberAccept(this);
        }
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        markCanNotBeMadePrivate(programField);
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        markCanNotBeMadePrivate(libraryField);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        markCanNotBeMadePrivate(programMethod);
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        markCanNotBeMadePrivate(libraryMethod);
    }


    // Small utility methods.

    private static void markCanNotBeMadePrivate(Field field)
    {
        FieldOptimizationInfo info = FieldOptimizationInfo.getFieldOptimizationInfo(field);
        if (info != null)
        {
            info.setCanNotBeMadePrivate();
        }
    }


    /**
     * Returns whether the given field can be made private.
     */
    public static boolean canBeMadePrivate(Field field)
    {
        FieldOptimizationInfo info = FieldOptimizationInfo.getFieldOptimizationInfo(field);
        return info != null &&
               info.canBeMadePrivate();
    }


    private static void markCanNotBeMadePrivate(Method method)
    {
        MethodOptimizationInfo info = MethodOptimizationInfo.getMethodOptimizationInfo(method);
        if (info != null)
        {
            info.setCanNotBeMadePrivate();
        }
    }


    /**
     * Returns whether the given method can be made private.
     */
    public static boolean canBeMadePrivate(Method method)
    {
        MethodOptimizationInfo info = MethodOptimizationInfo.getMethodOptimizationInfo(method);
        return info != null &&
               info.canBeMadePrivate();
    }
}
