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
 * This ClassVisitor removes specified interfaces from the classes and class
 * signatures that it visits.
 *
 * @author Eric Lafortune
 */
public class InterfaceDeleter
extends      SimplifiedVisitor
implements   ClassVisitor,
             AttributeVisitor
{
    private static final boolean DEBUG = false;


    private final boolean[] delete;


    /**
     * Creates a new InterfaceDeleter to remove the specified interfaces.
     * @param delete an array that corresponds to the interfaces of a class
     *               and that specifies the ones to be removed.
     */
    public InterfaceDeleter(boolean[] delete)
    {
        this.delete = delete;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        int[] interfaces      = programClass.u2interfaces;
        int   interfacesCount = programClass.u2interfacesCount;

        if (DEBUG)
        {
            System.out.println("InterfaceDeleter: "+programClass.getName()+" ("+interfacesCount+" interfaces)");
        }

        // Copy the interfaces that aren't deleted.
        int newInterfacesCount = 0;
        for (int index = 0; index < interfacesCount; index++)
        {
            if (DEBUG)
            {
                System.out.println("InterfaceDeleter:   "+(delete[index]?"- ":"+ ")+programClass.getInterfaceName(index));
            }

            if (!delete[index])
            {
                interfaces[newInterfacesCount++] = interfaces[index];
            }
        }

        // Update the signature.
        if (newInterfacesCount < interfacesCount)
        {
            programClass.u2interfacesCount = newInterfacesCount;

            programClass.attributesAccept(this);
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitSignatureAttribute(Clazz clazz, SignatureAttribute signatureAttribute)
    {
        Clazz[] referencedClasses = signatureAttribute.referencedClasses;
        if (referencedClasses != null)
        {
            // Process the generic definitions, superclass, and implemented
            // interfaces.
            InternalTypeEnumeration internalTypeEnumeration =
                new InternalTypeEnumeration(signatureAttribute.getSignature(clazz));

            // Recompose the signature types in a string buffer.
            StringBuffer newSignatureBuffer = new StringBuffer();

            // Also update the array with referenced classes.
            int referencedClassIndex    = 0;
            int newReferencedClassIndex = 0;

            // Copy the variable type declarations.
            if (internalTypeEnumeration.hasFormalTypeParameters())
            {
                String type = internalTypeEnumeration.formalTypeParameters();

                // Append the type.
                newSignatureBuffer.append(type);

                // Copy any referenced classes.
                int classCount =
                    new DescriptorClassEnumeration(type).classCount();

                for (int counter = 0; counter < classCount; counter++)
                {
                    referencedClasses[newReferencedClassIndex++] =
                        referencedClasses[referencedClassIndex++];
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
                int classCount =
                    new DescriptorClassEnumeration(type).classCount();

                for (int counter = 0; counter < classCount; counter++)
                {
                    referencedClasses[newReferencedClassIndex++] =
                        referencedClasses[referencedClassIndex++];
                }

                if (DEBUG)
                {
                    System.out.println("InterfaceDeleter:   super class type = " + type);
                }
            }

            // Copy the interface types.
            int index = 0;
            while (internalTypeEnumeration.hasMoreTypes())
            {
                String type = internalTypeEnumeration.nextType();

                int classCount =
                    new DescriptorClassEnumeration(type).classCount();

                if (DEBUG)
                {
                    System.out.println("InterfaceDeleter:   interface type " + (delete[index] ? "- " : "+ ") + type + " (" + classCount + " referenced classes)");
                }

                if (!delete[index++])
                {
                    // Append the type.
                    newSignatureBuffer.append(type);

                    // Copy any referenced classes.
                    for (int counter = 0; counter < classCount; counter++)
                    {
                        referencedClasses[newReferencedClassIndex++] =
                            referencedClasses[referencedClassIndex++];
                    }
                }
                else
                {
                    referencedClassIndex += classCount;
                }
            }

            // Update the signature.
            ((Utf8Constant)((ProgramClass)clazz).constantPool[signatureAttribute.u2signatureIndex]).setString(newSignatureBuffer.toString());

            // Clear the remaining referenced classes.
            Arrays.fill(referencedClasses,
                        newReferencedClassIndex,
                        referencedClassIndex,
                        null);
        }
    }
}
