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
package proguard;

import proguard.classfile.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;

import java.util.*;

/**
 * This class checks if the user has forgotten to fully qualify any classes
 * in the configuration.
 *
 * @author Eric Lafortune
 */
public class FullyQualifiedClassNameChecker
extends      SimplifiedVisitor
implements   ClassVisitor
{
    private static final String INVALID_CLASS_EXTENSION = ClassUtil.internalClassName(ClassConstants.CLASS_FILE_EXTENSION);


    private final ClassPool      programClassPool;
    private final ClassPool      libraryClassPool;
    private final WarningPrinter notePrinter;


    /**
     * Creates a new FullyQualifiedClassNameChecker.
     */
    public FullyQualifiedClassNameChecker(ClassPool      programClassPool,
                                          ClassPool      libraryClassPool,
                                          WarningPrinter notePrinter)
    {
        this.programClassPool = programClassPool;
        this.libraryClassPool = libraryClassPool;
        this.notePrinter      = notePrinter;
    }


    /**
     * Checks the classes mentioned in the given class specifications, printing
     * notes if necessary.
     */
    public void checkClassSpecifications(List classSpecifications)
    {
        if (classSpecifications != null)
        {
            for (int index = 0; index < classSpecifications.size(); index++)
            {
                ClassSpecification classSpecification =
                    (ClassSpecification)classSpecifications.get(index);

                checkType(classSpecification.annotationType);
                checkClassName(classSpecification.className);
                checkType(classSpecification.extendsAnnotationType);
                checkClassName(classSpecification.extendsClassName);

                checkMemberSpecifications(classSpecification.fieldSpecifications,  true);
                checkMemberSpecifications(classSpecification.methodSpecifications, false);
            }
        }
    }


    /**
     * Checks the classes mentioned in the given class member specifications,
     * printing notes if necessary.
     */
    private void checkMemberSpecifications(List memberSpecifications, boolean isField)
    {
        if (memberSpecifications != null)
        {
            for (int index = 0; index < memberSpecifications.size(); index++)
            {
                MemberSpecification memberSpecification =
                    (MemberSpecification)memberSpecifications.get(index);

                checkType(memberSpecification.annotationType);

                if (isField)
                {
                     checkType(memberSpecification.descriptor);
                }
                else
                {
                    checkDescriptor(memberSpecification.descriptor);
                }
            }
        }
    }


    /**
     * Checks the classes mentioned in the given class member descriptor,
     * printing notes if necessary.
     */
    private void checkDescriptor(String descriptor)
    {
        if (descriptor != null)
        {
            InternalTypeEnumeration internalTypeEnumeration =
                new InternalTypeEnumeration(descriptor);

            checkType(internalTypeEnumeration.returnType());

            while (internalTypeEnumeration.hasMoreTypes())
            {
                checkType(internalTypeEnumeration.nextType());
            }
        }
    }


    /**
     * Checks the class mentioned in the given type (if any),
     * printing notes if necessary.
     */
    private void checkType(String type)
    {
        if (type != null)
        {
            checkClassName(ClassUtil.internalClassNameFromType(type));
        }
    }


    /**
     * Checks the specified class (if any),
     * printing notes if necessary.
     */
    private void checkClassName(String className)
    {
        if (className != null                            &&
            !containsWildCards(className)                &&
            programClassPool.getClass(className) == null &&
            libraryClassPool.getClass(className) == null &&
            notePrinter.accepts(className))
        {
            notePrinter.print(className,
                              "Note: the configuration refers to the unknown class '" +
                              ClassUtil.externalClassName(className) + "'");

            // Strip "/class" or replace the package name by a wildcard.
            int lastSeparatorIndex =
                className.lastIndexOf(ClassConstants.PACKAGE_SEPARATOR);

            String fullyQualifiedClassName =
                className.endsWith(INVALID_CLASS_EXTENSION) ?
                    className.substring(0, lastSeparatorIndex) :
                    "**" + ClassConstants.PACKAGE_SEPARATOR + className.substring(lastSeparatorIndex + 1);

            // Suggest matching classes.
            ClassNameFilter classNameFilter =
                new ClassNameFilter(fullyQualifiedClassName, this);

            programClassPool.classesAccept(classNameFilter);
            libraryClassPool.classesAccept(classNameFilter);
        }
    }


    private static boolean containsWildCards(String string)
    {
        return string != null &&
            (string.indexOf('!')   >= 0 ||
             string.indexOf('*')   >= 0 ||
             string.indexOf('?')   >= 0 ||
             string.indexOf(',')   >= 0 ||
             string.indexOf("///") >= 0 ||
             string.indexOf('<')   >= 0);
    }


    // Implementations for ClassVisitor.

    public void visitAnyClass(Clazz clazz)
    {
        System.out.println("      Maybe you meant the fully qualified name '" +
                           ClassUtil.externalClassName(clazz.getName()) + "'?");
    }
}
