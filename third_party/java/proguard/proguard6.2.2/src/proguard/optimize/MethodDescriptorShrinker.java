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
 * This MemberVisitor removes unused parameters in the descriptors of the
 * methods that it visits. It also updates the signatures and parameter
 * annotations.
 *
 * @see ParameterUsageMarker
 * @see VariableUsageMarker
 * @see ParameterShrinker
 * @author Eric Lafortune
 */
public class MethodDescriptorShrinker
extends      SimplifiedVisitor
implements   MemberVisitor,
             AttributeVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("mds") != null;
    //*/


    private final MemberVisitor extraMemberVisitor;


    /**
     * Creates a new MethodDescriptorShrinker.
     */
    public MethodDescriptorShrinker()
    {
        this(null);
    }


    /**
     * Creates a new MethodDescriptorShrinker with an extra visitor.
     * @param extraMemberVisitor an optional extra visitor for all methods whose
     *                           parameters have been simplified.
     */
    public MethodDescriptorShrinker(MemberVisitor extraMemberVisitor)
    {
        this.extraMemberVisitor = extraMemberVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if (DEBUG)
        {
            System.out.println("MethodDescriptorShrinker: ["+programClass.getName()+"."+programMethod.getName(programClass)+programMethod.getDescriptor(programClass)+"]");
        }

        // Update the descriptor if it has any unused parameters.
        String descriptor    = programMethod.getDescriptor(programClass);
        String newDescriptor = shrinkDescriptor(programMethod, descriptor, 0);

        if (!newDescriptor.equals(descriptor))
        {
            // Shrink the signature and parameter annotations,
            // before shrinking the descriptor itself.
            programMethod.attributesAccept(programClass, this);

            String name    = programMethod.getName(programClass);
            String newName = name;

            // Append a code, if the method isn't a class instance initializer.
            if (!name.equals(ClassConstants.METHOD_NAME_INIT))
            {
                newName += ClassConstants.SPECIAL_MEMBER_SEPARATOR + Long.toHexString(Math.abs((descriptor).hashCode()));
            }

            ConstantPoolEditor constantPoolEditor =
                new ConstantPoolEditor(programClass);

            // Update the name, if necessary.
            if (!newName.equals(name))
            {
                programMethod.u2nameIndex =
                    constantPoolEditor.addUtf8Constant(newName);
            }

            // Update the referenced classes.
            programMethod.referencedClasses =
                shrinkReferencedClasses(programMethod,
                                        descriptor,
                                        0,
                                        programMethod.referencedClasses);

            // Finally, update the descriptor itself.
            programMethod.u2descriptorIndex =
                constantPoolEditor.addUtf8Constant(newDescriptor);

            if (DEBUG)
            {
                System.out.println("    -> ["+newName+newDescriptor+"]");
            }

            // Visit the method, if required.
            if (extraMemberVisitor != null)
            {
                extraMemberVisitor.visitProgramMethod(programClass, programMethod);
            }
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitSignatureAttribute(Clazz clazz, Method method, SignatureAttribute signatureAttribute)
    {
        if (DEBUG)
        {
            System.out.println("  ["+signatureAttribute.getSignature(clazz)+"]");
        }

        // Compute the new signature.
        String signature = signatureAttribute.getSignature(clazz);

        // Constructors of enum classes and of non-static inner classes may
        // start with one or two synthetic parameters, which are not part
        // of the signature.
        int syntheticParametersSize =
            new InternalTypeEnumeration(method.getDescriptor(clazz)).typesSize() -
            new InternalTypeEnumeration(signature).typesSize();

        String newSignature = shrinkDescriptor(method,
                                               signature,
                                               syntheticParametersSize);

        if (!newSignature.equals(signature))
        {
            // Update the signature.
            signatureAttribute.u2signatureIndex =
                new ConstantPoolEditor((ProgramClass)clazz).addUtf8Constant(newSignature);

            // Update the referenced classes.
            signatureAttribute.referencedClasses =
                shrinkReferencedClasses(method,
                                        signature,
                                        syntheticParametersSize,
                                        signatureAttribute.referencedClasses);

            if (DEBUG)
            {
                System.out.println("    -> ["+newSignature+"]");
            }
        }
    }


    public void visitAnyParameterAnnotationsAttribute(Clazz clazz, Method method, ParameterAnnotationsAttribute parameterAnnotationsAttribute)
    {
        int[]          annotationsCounts = parameterAnnotationsAttribute.u2parameterAnnotationsCount;
        Annotation[][] annotations       = parameterAnnotationsAttribute.parameterAnnotations;

        String descriptor = method.getDescriptor(clazz);

        // Constructors of enum classes and of non-static inner classes may
        // start with one or two synthetic parameters, which are not part
        // of the signature and not counted for parameter annotations.
        int syntheticParameterCount =
            new InternalTypeEnumeration(descriptor).typeCount() -
            parameterAnnotationsAttribute.u1parametersCount;

        int syntheticParametersSize =
            ClassUtil.internalMethodVariableIndex(descriptor,
                                                  true,
                                                  syntheticParameterCount);

        // All parameters of non-static methods are shifted by one in the local
        // variable frame.
        int parameterIndex =
            syntheticParametersSize +
            ((method.getAccessFlags() & ClassConstants.ACC_STATIC) != 0 ?
                 0 : 1);

        int annotationIndex    = 0;
        int newAnnotationIndex = 0;

        // Go over the parameters.
        InternalTypeEnumeration internalTypeEnumeration =
            new InternalTypeEnumeration(descriptor);

        // Skip synthetic parameters that don't have annotations.
        for (int counter = 0; counter < syntheticParameterCount; counter++)
        {
            internalTypeEnumeration.nextType();
        }

        // Shrink the annotations of the actual parameters.
        while (internalTypeEnumeration.hasMoreTypes())
        {
            String type = internalTypeEnumeration.nextType();
            if (ParameterUsageMarker.isParameterUsed(method, parameterIndex))
            {
                annotationsCounts[newAnnotationIndex] = annotationsCounts[annotationIndex];
                annotations[newAnnotationIndex++]     = annotations[annotationIndex];
            }

            annotationIndex++;

            parameterIndex += ClassUtil.internalTypeSize(type);
        }

        // Update the number of parameters.
        parameterAnnotationsAttribute.u1parametersCount = newAnnotationIndex;

        // Clear the unused entries.
        while (newAnnotationIndex < annotationIndex)
        {
            annotationsCounts[newAnnotationIndex] = 0;
            annotations[newAnnotationIndex++]     = null;
        }
    }


    // Small utility methods.

    /**
     * Returns a shrunk descriptor or signature of the given method.
     */
    private String shrinkDescriptor(Method method,
                                    String descriptor,
                                    int    syntheticParametersSize)
    {
        // Signatures only start after any synthetic parameters.
        // All parameters of non-static methods are shifted by one in the local
        // variable frame.
        int parameterIndex =
            syntheticParametersSize +
            ((method.getAccessFlags() & ClassConstants.ACC_STATIC) != 0 ?
                 0 : 1);

        InternalTypeEnumeration internalTypeEnumeration =
            new InternalTypeEnumeration(descriptor);

        StringBuffer newDescriptorBuffer =
            new StringBuffer(descriptor.length());

        // Copy the formal type parameters.
        newDescriptorBuffer.append(internalTypeEnumeration.formalTypeParameters());
        newDescriptorBuffer.append(ClassConstants.METHOD_ARGUMENTS_OPEN);

        // Go over the parameters.
        while (internalTypeEnumeration.hasMoreTypes())
        {
            String type = internalTypeEnumeration.nextType();
            if (ParameterUsageMarker.isParameterUsed(method, parameterIndex))
            {
                newDescriptorBuffer.append(type);
            }
            else if (DEBUG)
            {
                System.out.println("  Deleting parameter #"+parameterIndex+" ["+type+"]");
            }

            parameterIndex += ClassUtil.internalTypeSize(type);
        }

        // Copy the return type.
        newDescriptorBuffer.append(ClassConstants.METHOD_ARGUMENTS_CLOSE);
        newDescriptorBuffer.append(internalTypeEnumeration.returnType());

        return newDescriptorBuffer.toString();
    }


    /**
     * Shrinks the array of referenced classes of the given method.
     */
    private Clazz[] shrinkReferencedClasses(Method  method,
                                            String  descriptor,
                                            int     syntheticParametersSize,
                                            Clazz[] referencedClasses)
    {
        if (referencedClasses != null)
        {
            // Signatures only start after any synthetic parameters.
            // All parameters of non-static methods are shifted by one in the local
            // variable frame.
            int parameterIndex =
                syntheticParametersSize +
                ((method.getAccessFlags() & ClassConstants.ACC_STATIC) != 0 ?
                     0 : 1);

            InternalTypeEnumeration internalTypeEnumeration =
                new InternalTypeEnumeration(descriptor);

            int referencedClassIndex    = 0;
            int newReferencedClassIndex = 0;

            // Copy the formal type parameters.
            {
                String type = internalTypeEnumeration.formalTypeParameters();
                int count = new DescriptorClassEnumeration(type).classCount();
                for (int counter = 0; counter < count; counter++)
                {
                    referencedClasses[newReferencedClassIndex++] =
                        referencedClasses[referencedClassIndex++];
                }
            }

            // Go over the parameters.
            while (internalTypeEnumeration.hasMoreTypes())
            {
                // Consider the classes referenced by this parameter type.
                String type  = internalTypeEnumeration.nextType();
                int    count = new DescriptorClassEnumeration(type).classCount();

                if (ParameterUsageMarker.isParameterUsed(method, parameterIndex))
                {
                    // Copy the referenced classes.
                    for (int counter = 0; counter < count; counter++)
                    {
                        referencedClasses[newReferencedClassIndex++] =
                            referencedClasses[referencedClassIndex++];
                    }
                }
                else
                {
                    // Skip the referenced classes.
                    referencedClassIndex += count;
                }

                parameterIndex += ClassUtil.internalTypeSize(type);
            }

            // Copy the return type.
            {
                String type = internalTypeEnumeration.returnType();
                int count = new DescriptorClassEnumeration(type).classCount();
                for (int counter = 0; counter < count; counter++)
                {
                    referencedClasses[newReferencedClassIndex++] =
                        referencedClasses[referencedClassIndex++];
                }
            }

            // Shrink the array to the proper size.
            if (newReferencedClassIndex == 0)
            {
                referencedClasses = null;
            }
            else if (newReferencedClassIndex < referencedClassIndex)
            {
                Clazz[] newReferencedClasses = new Clazz[newReferencedClassIndex];
                System.arraycopy(referencedClasses, 0,
                                 newReferencedClasses, 0,
                                 newReferencedClassIndex);

                referencedClasses = newReferencedClasses;
            }
        }

        return referencedClasses;
    }
}
