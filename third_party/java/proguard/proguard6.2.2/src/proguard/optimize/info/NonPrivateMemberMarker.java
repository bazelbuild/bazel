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
package proguard.optimize.info;

import proguard.classfile.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;
import proguard.optimize.OptimizationInfoMemberFilter;

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
    private final MemberVisitor filteredMemberMarker    = new OptimizationInfoMemberFilter(this);
    private final MemberVisitor implementedMethodMarker = new OptimizationInfoMemberFilter(
                                                          new MethodImplementationFilter(this));


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Mark all referenced class members in different classes.
        programClass.constantPoolEntriesAccept(this);

        // Explicitly mark the <clinit> method.
        programClass.methodAccept(ClassConstants.METHOD_NAME_CLINIT,
                                  ClassConstants.METHOD_TYPE_CLINIT,
                                  filteredMemberMarker);

        // Explicitly mark the parameterless <init> method.
        programClass.methodAccept(ClassConstants.METHOD_NAME_INIT,
                                  ClassConstants.METHOD_TYPE_INIT,
                                  filteredMemberMarker);

        // Mark all methods that may have implementations.
        programClass.methodsAccept(implementedMethodMarker);
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        // The referenced class member, if any, can never be made private,
        // even if it's in the same class.
        stringConstant.referencedMemberAccept(filteredMemberMarker);
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
            refConstant.referencedMemberAccept(filteredMemberMarker);
        }
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        markCanNotBeMadePrivate(programField);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        markCanNotBeMadePrivate(programMethod);
    }


    // Small utility methods.

    private static void markCanNotBeMadePrivate(Field field)
    {
        ProgramFieldOptimizationInfo.getProgramFieldOptimizationInfo(field).setCanNotBeMadePrivate();
    }


    /**
     * Returns whether the given field can be made private.
     */
    public static boolean canBeMadePrivate(Field field)
    {
        return FieldOptimizationInfo.getFieldOptimizationInfo(field).canBeMadePrivate();
    }


    private static void markCanNotBeMadePrivate(Method method)
    {
        ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method).setCanNotBeMadePrivate();
    }


    /**
     * Returns whether the given method can be made private.
     */
    public static boolean canBeMadePrivate(Method method)
    {
        return MethodOptimizationInfo.getMethodOptimizationInfo(method).canBeMadePrivate();
    }
}
