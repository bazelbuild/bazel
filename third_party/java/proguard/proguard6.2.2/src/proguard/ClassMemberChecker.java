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
import proguard.classfile.visitor.MemberVisitor;

import java.util.List;

/**
 * This class checks if the user has specified non-existent class members.
 *
 * @author Eric Lafortune
 */
public class ClassMemberChecker
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final ClassPool      programClassPool;
    private final WarningPrinter notePrinter;


    /**
     * Creates a new ClassMemberChecker.
     */
    public ClassMemberChecker(ClassPool      programClassPool,
                              WarningPrinter notePrinter)
    {
        this.programClassPool = programClassPool;
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

                String className = classSpecification.className;
                if (className != null             &&
                    !containsWildCards(className) &&
                    notePrinter.accepts(className))
                {
                    Clazz clazz = programClassPool.getClass(className);
                    if (clazz != null)
                    {
                        checkMemberSpecifications(clazz, classSpecification.fieldSpecifications,  true);
                        checkMemberSpecifications(clazz, classSpecification.methodSpecifications, false);
                    }
                }
            }
        }
    }


    /**
     * Checks the class members mentioned in the given class member
     * specifications, printing notes if necessary.
     */
    private void checkMemberSpecifications(Clazz   clazz,
                                           List    memberSpecifications,
                                           boolean isField)
    {
        if (memberSpecifications != null)
        {
            String className = clazz.getName();

            for (int index = 0; index < memberSpecifications.size(); index++)
            {
                MemberSpecification memberSpecification =
                    (MemberSpecification)memberSpecifications.get(index);

                String memberName = memberSpecification.name;
                String descriptor = memberSpecification.descriptor;
                if (memberName != null             &&
                    !containsWildCards(memberName) &&
                    descriptor != null             &&
                    !containsWildCards(descriptor))
                {
                    if (isField)
                    {
                        if (clazz.findField(memberName, descriptor) == null)
                        {
                            notePrinter.print(className,
                                              "Note: the configuration refers to the unknown field '" +
                                              ClassUtil.externalFullFieldDescription(0, memberName, descriptor) + "' in class '" +
                                              ClassUtil.externalClassName(className) + "'");
                        }
                    }
                    else
                    {
                        if (clazz.findMethod(memberName, descriptor) == null)
                        {
                            notePrinter.print(className,
                                              "Note: the configuration refers to the unknown method '" +
                                              ClassUtil.externalFullMethodDescription(className, 0, memberName, descriptor) + "' in class '" +
                                              ClassUtil.externalClassName(className) + "'");
                        }
                    }
                }
            }
        }
    }


    private static boolean containsWildCards(String string)
    {
        return string != null &&
            (string.indexOf('!')   >= 0 ||
             string.indexOf('*')   >= 0 ||
             string.indexOf('?')   >= 0 ||
             string.indexOf(',')   >= 0 ||
             string.indexOf("///") >= 0);
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        System.out.println("      Maybe you meant the field '" +
                           ClassUtil.externalFullFieldDescription(0, programField.getName(programClass), programField.getDescriptor(programClass)) + "'?");
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        System.out.println("      Maybe you meant the method '" +
                           ClassUtil.externalFullMethodDescription(programClass.getName(), 0, programMethod.getName(programClass), programMethod.getDescriptor(programClass)) + "'?");
    }
}