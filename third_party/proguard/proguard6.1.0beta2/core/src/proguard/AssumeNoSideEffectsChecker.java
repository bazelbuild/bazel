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

import java.util.List;

/**
 * This class checks if the user is specifying to assume no side effects
 * for a reasonable number of methods in a class: not none and not all.
 *
 * @author Eric Lafortune
 */
public class AssumeNoSideEffectsChecker
{
    private final WarningPrinter notePrinter;


    /**
     * Creates a new KeepClassMemberChecker.
     */
    public AssumeNoSideEffectsChecker(WarningPrinter notePrinter)
    {
        this.notePrinter = notePrinter;
    }


    /**
     * Checks if the given class specifications try to assume no side effects
     * for all methods in a class, printing notes if necessary.
     */
    public void checkClassSpecifications(List classSpecifications)
    {
        if (classSpecifications != null)
        {
            for (int classSpecificationIndex = 0;
                 classSpecificationIndex < classSpecifications.size();
                 classSpecificationIndex++)
            {
                ClassSpecification classSpecification =
                    (ClassSpecification)classSpecifications.get(classSpecificationIndex);

                String className = classSpecification.className;
                if (className == null)
                {
                    className = classSpecification.extendsClassName;
                }

                if (className == null ||
                    notePrinter.accepts(className))
                {
                    List methodSpecifications =
                        classSpecification.methodSpecifications;

                    if (methodSpecifications != null)
                    {
                        for (int methodSpecificationIndex = 0;
                             methodSpecificationIndex < methodSpecifications.size();
                             methodSpecificationIndex++)
                        {
                            final MemberSpecification methodSpecification =
                                (MemberSpecification)methodSpecifications.get(methodSpecificationIndex);

                            if (methodSpecification.name       == null &&
                                methodSpecification.descriptor == null)
                            {
                                notePrinter.print(className,
                                                  "Note: the configuration specifies that none of the methods of class '" +
                                                  (className == null ?
                                                       ConfigurationConstants.ANY_CLASS_KEYWORD :
                                                       ClassUtil.externalClassName(className)) + "' have any side effects");
                            }
                        }
                    }
                }
            }
        }
    }
}
