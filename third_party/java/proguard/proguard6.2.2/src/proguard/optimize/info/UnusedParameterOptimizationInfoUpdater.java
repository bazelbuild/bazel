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
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.optimize.*;

/**
 * This AttributeVisitor removes unused parameters from the optimization info
 * of the methods that it visits. This includes 'this' parameters.
 *
 * @see ParameterUsageMarker
 * @see MethodStaticizer
 * @see MethodDescriptorShrinker
 * @author Eric Lafortune
 */
public class UnusedParameterOptimizationInfoUpdater
extends      SimplifiedVisitor
implements   AttributeVisitor,

             // Internal implementations.
             ParameterVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("upoiu") != null;
    //*/


    private final MemberVisitor extraUnusedParameterMethodVisitor;

    private final MemberVisitor unusedParameterRemover = new AllParameterVisitor(true,
                                                         new UsedParameterFilter(null, this));

    // Parameters and return values for visitor methods.
    private int removedParameterSize;
    private int removedParameterCount;


    /**
     * Creates a new UnusedParameterOptimizationInfoUpdater.
     */
    public UnusedParameterOptimizationInfoUpdater()
    {
        this(null);
    }


    /**
     * Creates a new UnusedParameterOptimizationInfoUpdater with an extra
     * visitor.
     * @param extraUnusedParameterMethodVisitor an optional extra visitor for
     *                                          all removed parameters.
     */
    public UnusedParameterOptimizationInfoUpdater(MemberVisitor extraUnusedParameterMethodVisitor)
    {
        this.extraUnusedParameterMethodVisitor = extraUnusedParameterMethodVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        if (DEBUG)
        {
            System.out.println("UnusedParameterOptimizationInfoUpdater: "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz));
        }

        // Update the optimization info.
        removedParameterCount = 0;
        removedParameterSize  = 0;

        method.accept(clazz, unusedParameterRemover);

        // Compute the new parameter size from the shrunk descriptor.
        ProgramMethodOptimizationInfo programMethodOptimizationInfo =
            ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method);

        int newParameterSize =
            programMethodOptimizationInfo.getParameterSize() - removedParameterSize;

        programMethodOptimizationInfo.setParameterSize(newParameterSize);
        programMethodOptimizationInfo.updateUsedParameters(-1L);
    }


    // Implementations for ParameterVisitor.

    public void visitParameter(Clazz clazz, Member member, int parameterIndex, int parameterCount, int parameterOffset, int parameterSize, String parameterType, Clazz referencedClass)
    {
        if (DEBUG)
        {
            System.out.println("  Deleting parameter #"+parameterIndex+" (v"+parameterOffset+")");
        }

        Method method = (Method)member;

        // Remove the unused parameter in the optimization info.
        // Take into acount the delta from previously removed parameters.
        ProgramMethodOptimizationInfo programMethodOptimizationInfo =
            ProgramMethodOptimizationInfo.getProgramMethodOptimizationInfo(method);

        programMethodOptimizationInfo.removeParameter(parameterIndex - removedParameterCount++);

        removedParameterSize += ClassUtil.internalTypeSize(parameterType);

        // Visit the method, if required.
        if (extraUnusedParameterMethodVisitor != null)
        {
            method.accept(clazz, extraUnusedParameterMethodVisitor);
        }
    }
}
