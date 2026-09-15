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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.constant.Utf8Constant;
import proguard.classfile.util.*;
import proguard.classfile.visitor.ClassVisitor;

import java.util.Arrays;

/**
 * This ClassVisitor sorts the interfaces of the program classes that it visits.
 *
 * @author Eric Lafortune
 */
public class InterfaceSorter
extends      SimplifiedVisitor
implements   ClassVisitor,
             AttributeVisitor
{
    private static final boolean DEBUG = false;


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        int[] interfaces      = programClass.u2interfaces;
        int   interfacesCount = programClass.u2interfacesCount;

        if (interfacesCount > 1)
        {
            // Sort the interfaces.
            Arrays.sort(interfaces, 0, interfacesCount);

            // Update the signature.
            programClass.attributesAccept(this);

            // Remove any duplicate entries.
            boolean[] delete = null;
            for (int index = 1; index < interfacesCount; index++)
            {
                if (interfaces[index] == interfaces[index - 1])
                {
                    // Lazily create the array.
                    if (delete == null)
                    {
                        delete = new boolean[interfacesCount];
                    }

                    delete[index] = true;
                }
            }

            if (delete != null)
            {
                new InterfaceDeleter(delete).visitProgramClass(programClass);
            }
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        Clazz[] referencedClasses    = signatureAttribute.referencedClasses;
        Clazz[] newReferencedClasses = referencedClasses == null ? null :
            new Clazz[referencedClasses.length];

        // Recompose the signature types in a string buffer.
        StringBuffer newSignatureBuffer = new StringBuffer();

        // Also update the array with referenced classes.
        int referencedClassIndex    = 0;
        int newReferencedClassIndex = 0;

        // Process the generic definitions and superclass.
        InternalTypeEnumeration internalTypeEnumeration =
            new InternalTypeEnumeration(signatureAttribute.getSignature(clazz));

        // Copy the variable type declarations.
        if (internalTypeEnumeration.hasFormalTypeParameters())
        {
            String type = internalTypeEnumeration.formalTypeParameters();

            // Append the type.
            newSignatureBuffer.append(type);

            // Copy any referenced classes.
            if (newReferencedClasses != null)
            {
                int classCount =
                    new DescriptorClassEnumeration(type).classCount();

                for (int counter = 0; counter < classCount; counter++)
                {
                    newReferencedClasses[newReferencedClassIndex++] =
                        referencedClasses[referencedClassIndex++];
                }
            }

            if (DEBUG)
            {
                System.out.println("InterfaceDeleter:   type parameters = " + type);
            }
        }

        // Copy the super class type.
        if (internalTypeEnumeration.hasMoreTypes())
        {
            String type = internalTypeEnumeration.nextType();

            // Append the type.
            newSignatureBuffer.append(type);

            // Copy any referenced classes.
            if (newReferencedClasses != null)
            {
                int classCount =
                    new DescriptorClassEnumeration(type).classCount();

                for (int counter = 0; counter < classCount; counter++)
                {
                    newReferencedClasses[newReferencedClassIndex++] =
                        referencedClasses[referencedClassIndex++];
                }
            }

            if (DEBUG)
            {
                System.out.println("InterfaceSorter:   super class type = " + type);
            }
        }

        int firstReferencedInterfaceIndex = referencedClassIndex;

        // Copy the interface types, based on the sorted interface classes.
        // This has the advantage that we will disregard any interface types
        // that are not in the interface classes, like in some versions of
        // the Scala runtime library.
        for (int interfaceIndex = 0; interfaceIndex < clazz.getInterfaceCount(); interfaceIndex++)
        {
            // Consider the interface class name.
            String interfaceName = clazz.getInterfaceName(interfaceIndex);

            referencedClassIndex = firstReferencedInterfaceIndex;

            // Find the corresponding interface type.
            InternalTypeEnumeration internalInterfaceTypeEnumeration =
                new InternalTypeEnumeration(signatureAttribute.getSignature(clazz));

            // Skip the superclass type.
            internalInterfaceTypeEnumeration.nextType();

            while (internalInterfaceTypeEnumeration.hasMoreTypes())
            {
                String type = internalInterfaceTypeEnumeration.nextType();

                DescriptorClassEnumeration classEnumeration =
                    new DescriptorClassEnumeration(type);

                int classCount =
                    classEnumeration.classCount();

                classEnumeration.nextFluff();

                if (interfaceName.equals(classEnumeration.nextClassName()))
                {
                    // Append the type.
                    newSignatureBuffer.append(type);

                    // Copy any referenced classes.
                    if (newReferencedClasses != null)
                    {
                        for (int counter = 0; counter < classCount; counter++)
                        {
                            newReferencedClasses[newReferencedClassIndex++] =
                                referencedClasses[referencedClassIndex++];
                        }
                    }

                    if (DEBUG)
                    {
                        System.out.println("InterfaceSorter:   interface type = " + type);
                    }
                }
                else
                {
                    // Skip all referenced classes.
                    referencedClassIndex += classCount;
                }
            }
        }

        String newSignature = newSignatureBuffer.toString();

        // Did the signature change?
        if (!newSignature.equals(signatureAttribute.getSignature(clazz)))
        {
            // Update the signature.
            ((Utf8Constant)((ProgramClass)clazz).constantPool[signatureAttribute.u2signatureIndex]).setString(newSignatureBuffer.toString());

            // Update the referenced classes.
            signatureAttribute.referencedClasses = newReferencedClasses;

            if (DEBUG)
            {
                System.out.println("InterfaceSorter: result = "+newSignature);
                System.out.println("InterfaceSorter: referenced classes:");

                if (newReferencedClasses != null)
                {
                    for (int index = 0; index < newReferencedClasses.length; index++)
                    {
                        System.out.println("  #"+index+" "+newReferencedClasses[index]);
                    }
                }
            }
        }
    }
}
