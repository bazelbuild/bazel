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
import proguard.classfile.editor.ClassReferenceFixer;
import proguard.classfile.util.*;
import proguard.classfile.visitor.MemberVisitor;
import proguard.evaluation.value.Value;
import proguard.optimize.evaluation.StoringInvocationUnit;

/**
 * This MemberVisitor specializes parameters in the descriptors of the
 * methods that it visits.
 *
 * @see StoringInvocationUnit
 * @see ClassReferenceFixer
 * @author Eric Lafortune
 */
public class MemberDescriptorSpecializer
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private static final boolean DEBUG = false;


    private final MemberVisitor extraParameterMemberVisitor;


    /**
     * Creates a new MethodDescriptorShrinker.
     */
    public MemberDescriptorSpecializer()
    {
        this(null);
    }


    /**
     * Creates a new MethodDescriptorShrinker with an extra visitor.
     * @param extraParameterMemberVisitor an optional extra visitor for all
     *                                    class members whose parameters have
     *                                    been specialized.
     */
    public MemberDescriptorSpecializer(MemberVisitor extraParameterMemberVisitor)
    {
        this.extraParameterMemberVisitor = extraParameterMemberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        Value parameterValue = StoringInvocationUnit.getFieldValue(programField);
        if (parameterValue.computationalType() == Value.TYPE_REFERENCE)
        {
            Clazz referencedClass = parameterValue.referenceValue().getReferencedClass();
            if (programField.referencedClass != referencedClass)
            {
                if (DEBUG)
                {
                    System.out.println("MemberDescriptorSpecializer: "+programClass.getName()+"."+programField.getName(programClass)+" "+programField.getDescriptor(programClass));
                    System.out.println("  "+programField.referencedClass.getName()+" -> "+referencedClass.getName());
                }

                programField.referencedClass = referencedClass;

                // Visit the field, if required.
                if (extraParameterMemberVisitor != null)
                {
                    extraParameterMemberVisitor.visitProgramField(programClass, programField);
                }
            }
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        // All parameters of non-static methods are shifted by one in the local
        // variable frame.
        boolean isStatic =
            (programMethod.getAccessFlags() & ClassConstants.ACC_STATIC) != 0;

        int parameterStart = isStatic ? 0 : 1;
        int parameterCount =
            ClassUtil.internalMethodParameterCount(programMethod.getDescriptor(programClass),
                                                   isStatic);

        int classIndex = 0;

        // Go over the parameters.
        for (int parameterIndex = parameterStart; parameterIndex < parameterCount; parameterIndex++)
        {
            Value parameterValue = StoringInvocationUnit.getMethodParameterValue(programMethod, parameterIndex);
             if (parameterValue.computationalType() == Value.TYPE_REFERENCE)
             {
                 Clazz referencedClass = parameterValue.referenceValue().getReferencedClass();
                 if (programMethod.referencedClasses[classIndex] != referencedClass)
                 {
                     if (DEBUG)
                     {
                         System.out.println("MemberDescriptorSpecializer: "+programClass.getName()+"."+programMethod.getName(programClass)+programMethod.getDescriptor(programClass));
                         System.out.println("  "+programMethod.referencedClasses[classIndex].getName()+" -> "+referencedClass.getName());
                     }

                     programMethod.referencedClasses[classIndex] = referencedClass;

                     // Visit the method, if required.
                     if (extraParameterMemberVisitor != null)
                     {
                         extraParameterMemberVisitor.visitProgramMethod(programClass, programMethod);
                     }
                 }

                 classIndex++;
             }
        }
    }
}
