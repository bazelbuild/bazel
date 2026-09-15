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
package proguard.classfile.util;

import proguard.classfile.*;
import proguard.classfile.visitor.*;

/**
 * This MemberVisitor lets a given parameter visitor visit all the parameters
 * of the methods that it visits. The parameters optionally includes the
 * 'this' parameter of non-static methods, but never the return value.
 *
 * @author Eric Lafortune
 */
public class AllParameterVisitor
implements   MemberVisitor
{
    private final boolean          includeThisParameter;
    private final ParameterVisitor parameterVisitor;


    /**
     * Creates a new AllParameterVisitor for the given parameter
     * visitor.
     */
    public AllParameterVisitor(boolean          includeThisParameter,
                               ParameterVisitor parameterVisitor)
    {
        this.includeThisParameter = includeThisParameter;
        this.parameterVisitor     = parameterVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        visitFieldType(programClass,
                       programField,
                       programField.referencedClass);
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        visitFieldType(libraryClass,
                       libraryField,
                       libraryField.referencedClass);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        visitParameters(programClass,
                        programMethod,
                        programMethod.referencedClasses);
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        visitParameters(libraryClass,
                        libraryMethod,
                        libraryMethod.referencedClasses);
    }


    // Small utility methods.

    /**
     * Lets the parameter visitor visit the type of the given field.
     */
    private void visitFieldType(Clazz clazz,
                                Field field,
                                Clazz referencedClass)
    {
        String descriptor = field.getDescriptor(clazz);
        parameterVisitor.visitParameter(clazz,
                                        field,
                                        0,
                                        1,
                                        0,
                                        ClassUtil.internalTypeSize(descriptor),
                                        descriptor,
                                        referencedClass);
    }


    /**
     * Lets the parameter visitor visit the parameters of the given method.
     */
    private void visitParameters(Clazz   clazz,
                                 Method  method,
                                 Clazz[] referencedClasses)
    {
        String descriptor = method.getDescriptor(clazz);

        // Count the number of parameters and their total size.
        int parameterCount  = 0;
        int parameterSize   = 0;

        int index = 1;

        loop: while (true)
        {
            char c = descriptor.charAt(index++);
            switch (c)
            {
                case ClassConstants.TYPE_LONG:
                case ClassConstants.TYPE_DOUBLE:
                {
                    // Long and double primitive types.
                    parameterSize++;
                    break;
                }
                default:
                {
                    // All other primitive types.
                    break;
                }
                case ClassConstants.TYPE_CLASS_START:
                {
                    // Class types.
                    // Skip the class name.
                    index = descriptor.indexOf(ClassConstants.TYPE_CLASS_END, index) + 1;
                    break;
                }
                case ClassConstants.TYPE_ARRAY:
                {
                    // Array types.
                    // Skip all array characters.
                    while ((c = descriptor.charAt(index++)) == ClassConstants.TYPE_ARRAY) {}

                    if (c == ClassConstants.TYPE_CLASS_START)
                    {
                        // Skip the class type.
                        index = descriptor.indexOf(ClassConstants.TYPE_CLASS_END, index) + 1;
                    }
                    break;
                }
                case ClassConstants.METHOD_ARGUMENTS_CLOSE:
                {
                    break loop;
                }
            }

            parameterCount++;
            parameterSize++;
        }

        // Visit the parameters.
        int parameterIndex  = 0;
        int parameterOffset = 0;
        int referenceClassIndex = 0;

        // Visit the 'this' parameter if applicable.
        if (includeThisParameter &&
            (method.getAccessFlags() & ClassConstants.ACC_STATIC) == 0)
        {
            parameterVisitor.visitParameter(clazz,
                                            method,
                                            parameterIndex++,
                                            ++parameterCount,
                                            parameterOffset++,
                                            ++parameterSize,
                                            ClassUtil.internalTypeFromClassName(clazz.getName()),
                                            clazz);
        }

        index = 1;

        while (true)
        {
            int    newIndex          = index + 1;
            int    thisParameterSize = 1;
            Clazz  referencedClass   = null;

            char c = descriptor.charAt(index);
            switch (c)
            {
                case ClassConstants.TYPE_LONG:
                case ClassConstants.TYPE_DOUBLE:
                {
                    // Long and double primitive types.
                    thisParameterSize = 2;
                    break;
                }
                default:
                {
                    // All other primitive types.
                    break;
                }
                case ClassConstants.TYPE_CLASS_START:
                {
                    // Class types.
                    // Skip the class name.
                    newIndex = descriptor.indexOf(ClassConstants.TYPE_CLASS_END, newIndex) + 1;
                    referencedClass = referencedClasses == null ? null :
                        referencedClasses[referenceClassIndex++];
                    break;
                }
                case ClassConstants.TYPE_ARRAY:
                {
                    // Array types.
                    // Skip all array characters.
                    while ((c = descriptor.charAt(newIndex++)) == ClassConstants.TYPE_ARRAY) {}

                    if (c == ClassConstants.TYPE_CLASS_START)
                    {
                        // Skip the class type.
                        newIndex = descriptor.indexOf(ClassConstants.TYPE_CLASS_END, newIndex) + 1;
                        referencedClass = referencedClasses == null ? null :
                            referencedClasses[referenceClassIndex++];
                    }
                    break;
                }
                case ClassConstants.METHOD_ARGUMENTS_CLOSE:
                {
                    // End of the method parameters.
                    return;
                }
            }

            parameterVisitor.visitParameter(clazz,
                                            method,
                                            parameterIndex++,
                                            parameterCount,
                                            parameterOffset,
                                            parameterSize,
                                            descriptor.substring(index, newIndex),
                                            referencedClass);

            // Continue with the next parameter.
            index = newIndex;
            parameterOffset += thisParameterSize;
        }
    }
}
