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
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.editor.ConstantPoolEditor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.MemberVisitor;
import proguard.optimize.info.*;

/**
 * This MemberVisitor adds an additional parameter to the duplicate
 * initialization methods that it visits.
 */
public class DuplicateInitializerFixer
extends      SimplifiedVisitor
implements   MemberVisitor,
             AttributeVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("dif") != null;
    //*/

    private static final char[] TYPES = new char[]
    {
        ClassConstants.TYPE_BYTE,
        ClassConstants.TYPE_CHAR,
        ClassConstants.TYPE_SHORT,
        ClassConstants.TYPE_INT,
        ClassConstants.TYPE_BOOLEAN
    };


    private final MemberVisitor extraFixedInitializerVisitor;


    /**
     * Creates a new DuplicateInitializerFixer.
     */
    public DuplicateInitializerFixer()
    {
        this(null);
    }


    /**
     * Creates a new DuplicateInitializerFixer with an extra visitor.
     * @param extraFixedInitializerVisitor an optional extra visitor for all
     *                                     initializers that have been fixed.
     */
    public DuplicateInitializerFixer(MemberVisitor extraFixedInitializerVisitor)
    {
        this.extraFixedInitializerVisitor = extraFixedInitializerVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        // Is it a class instance initializer?
        String name = programMethod.getName(programClass);
        if (name.equals(ClassConstants.METHOD_NAME_INIT))
        {
            // Is there already another initializer with the same descriptor?
            String descriptor    = programMethod.getDescriptor(programClass);
            Method similarMethod = programClass.findMethod(name, descriptor);
            if (!programMethod.equals(similarMethod))
            {
                // Should this initializer be preserved?
                if (KeepMarker.isKept(programMethod))
                {
                    // Fix the other initializer.
                    // We'll just proceed if it is being kept as well;
                    // apparently the descriptor types didn't matter so much.
                    programMethod = (ProgramMethod)similarMethod;
                }

                int index = descriptor.indexOf(ClassConstants.METHOD_ARGUMENTS_CLOSE);

                // Try to find a new, unique descriptor.
                int typeCounter = 0;
                while (true)
                {
                    // Construct the new descriptor by inserting a new type
                    // as an additional last argument.
                    StringBuffer newDescriptorBuffer =
                        new StringBuffer(descriptor.substring(0, index));

                    for (int arrayDimension = 0; arrayDimension < typeCounter / TYPES.length; arrayDimension++)
                    {
                        newDescriptorBuffer.append(ClassConstants.TYPE_ARRAY);
                    }

                    newDescriptorBuffer.append(TYPES[typeCounter % TYPES.length]);
                    newDescriptorBuffer.append(descriptor.substring(index));

                    String newDescriptor = newDescriptorBuffer.toString();

                    // Is the new initializer descriptor unique?
                    if (programClass.findMethod(name, newDescriptor) == null)
                    {
                        if (DEBUG)
                        {
                            System.out.println("DuplicateInitializerFixer:");
                            System.out.println("  ["+programClass.getName()+"."+name+descriptor+"] ("+ClassUtil.externalClassAccessFlags(programMethod.getAccessFlags())+") -> ["+newDescriptor+"]");
                        }

                        // Update the descriptor.
                        programMethod.u2descriptorIndex =
                            new ConstantPoolEditor(programClass).addUtf8Constant(newDescriptor);

                        // Fix the local variable frame size, the method
                        // signature, and the parameter annotations, if
                        // necessary.
                        programMethod.attributesAccept(programClass,
                                                       this);

                        // Update the optimization info.
                        MethodOptimizationInfo methodOptimizationInfo =
                            ProgramMethodOptimizationInfo.getMethodOptimizationInfo(programMethod);
                        if (methodOptimizationInfo instanceof ProgramMethodOptimizationInfo)
                        {
                            ProgramMethodOptimizationInfo programMethodOptimizationInfo =
                                (ProgramMethodOptimizationInfo)methodOptimizationInfo;

                            int parameterCount =
                                ClassUtil.internalMethodParameterCount(newDescriptor,
                                                                       programMethod.getAccessFlags());
                            programMethodOptimizationInfo.insertParameter(parameterCount - 1);

                            int parameterSize =
                                programMethodOptimizationInfo.getParameterSize();
                            programMethodOptimizationInfo.setParameterSize(parameterSize + 1);
                            programMethodOptimizationInfo.setParameterUsed(parameterSize);
                        }

                        // Visit the initializer, if required.
                        if (extraFixedInitializerVisitor != null)
                        {
                            extraFixedInitializerVisitor.visitProgramMethod(programClass, programMethod);
                        }

                        // We're done with this constructor.
                        return;
                    }

                    typeCounter++;
                }
            }
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // The minimum variable size is determined by the arguments.
        int maxLocals =
            ClassUtil.internalMethodParameterSize(method.getDescriptor(clazz),
                                                  method.getAccessFlags());

        if (codeAttribute.u2maxLocals < maxLocals)
        {
            codeAttribute.u2maxLocals = maxLocals;
        }
    }


    public void visitSignatureAttribute(Clazz clazz, Method method, SignatureAttribute signatureAttribute)
    {
        String descriptor      = method.getDescriptor(clazz);
        int    descriptorIndex = descriptor.indexOf(ClassConstants.METHOD_ARGUMENTS_CLOSE);
        String signature       = signatureAttribute.getSignature(clazz);
        int    signatureIndex  = signature.indexOf(ClassConstants.METHOD_ARGUMENTS_CLOSE);

        String newSignature = signature.substring(0, signatureIndex) +
                              descriptor.charAt(descriptorIndex - 1) +
                              signature.substring(signatureIndex);

        // Update the signature.
        signatureAttribute.u2signatureIndex =
            new ConstantPoolEditor((ProgramClass)clazz).addUtf8Constant(newSignature);
    }


    public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
    {
        // Update the number of parameters.
        int oldParametersCount = parameterAnnotationsAttribute.u1parametersCount++;

        if (parameterAnnotationsAttribute.u2parameterAnnotationsCount == null ||
            parameterAnnotationsAttribute.u2parameterAnnotationsCount.length < parameterAnnotationsAttribute.u1parametersCount)
        {
            int[]          annotationsCounts = new int[parameterAnnotationsAttribute.u1parametersCount];
            Annotation[][] annotations       = new Annotation[parameterAnnotationsAttribute.u1parametersCount][];

            System.arraycopy(parameterAnnotationsAttribute.u2parameterAnnotationsCount,
                             0,
                             annotationsCounts,
                             0,
                             oldParametersCount);

            System.arraycopy(parameterAnnotationsAttribute.parameterAnnotations,
                             0,
                             annotations,
                             0,
                             oldParametersCount);

            parameterAnnotationsAttribute.u2parameterAnnotationsCount = annotationsCounts;
            parameterAnnotationsAttribute.parameterAnnotations        = annotations;
        }
    }
}
