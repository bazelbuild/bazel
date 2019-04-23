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

import proguard.classfile.util.*;
import proguard.classfile.visitor.ClassVisitor;

import java.util.List;

/**
 * This class checks if the user has forgotten to specify class members in
 * some keep options in the configuration.
 *
 * @author Eric Lafortune
 */
public class KeepClassMemberChecker
{
    private final WarningPrinter notePrinter;


    /**
     * Creates a new KeepClassMemberChecker.
     */
    public KeepClassMemberChecker(WarningPrinter notePrinter)
    {
        this.notePrinter = notePrinter;
    }


    /**
     * Checks if the given class specifications try to keep class members
     * without actually specifying any, printing notes if necessary.
     */
    public void checkClassSpecifications(List keepClassSpecifications)
    {
        if (keepClassSpecifications != null)
        {
            for (int index = 0; index < keepClassSpecifications.size(); index++)
            {
                KeepClassSpecification keepClassSpecification =
                    (KeepClassSpecification)keepClassSpecifications.get(index);

                if (!keepClassSpecification.markClasses                      &&
                    (keepClassSpecification.fieldSpecifications  == null ||
                     keepClassSpecification.fieldSpecifications.size() == 0) &&
                    (keepClassSpecification.methodSpecifications == null ||
                     keepClassSpecification.methodSpecifications.size() == 0))
                {
                    String className = keepClassSpecification.className;
                    if (className == null)
                    {
                        className = keepClassSpecification.extendsClassName;
                    }

                    if (className == null ||
                        notePrinter.accepts(className))
                    {
                        notePrinter.print(className,
                                          "Note: the configuration doesn't specify which class members to keep for class '" +
                                          (className == null ?
                                              ConfigurationConstants.ANY_CLASS_KEYWORD :
                                              ClassUtil.externalClassName(className)) + "'");
                    }
                }
            }
        }
    }
}
