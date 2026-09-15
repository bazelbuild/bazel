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

import java.util.List;

/**
 * This class checks whether some keep rules only keep library classes, no
 * program classes. That is strange, because library classes never need to
 * be kept explicitly.
 *
 * @author Eric Lafortune
 */
public class LibraryKeepChecker
implements   ClassVisitor
{
    private final ClassPool      programClassPool;
    private final ClassPool      libraryClassPool;
    private final WarningPrinter notePrinter;

    // Some fields acting as parameters for the class visitor.
    private String keepName;


    /**
     * Creates a new DescriptorKeepChecker.
     */
    public LibraryKeepChecker(ClassPool      programClassPool,
                              ClassPool      libraryClassPool,
                              WarningPrinter notePrinter)
    {
        this.programClassPool = programClassPool;
        this.libraryClassPool = libraryClassPool;
        this.notePrinter      = notePrinter;
    }


    /**
     * Checks the classes mentioned in the given keep specifications, printing
     * notes if necessary.
     */
    public void checkClassSpecifications(List keepSpecifications)
    {
        if (keepSpecifications != null)
        {
            // Go over all individual keep specifications.
            for (int index = 0; index < keepSpecifications.size(); index++)
            {
                KeepClassSpecification keepClassSpecification =
                    (KeepClassSpecification)keepSpecifications.get(index);

                // Is the keep specification more specific than a general
                // wildcard?
                keepName = keepClassSpecification.className;
                if (keepName != null)
                {
                    KeepClassSpecificationVisitorFactory visitorFactory =
                        new KeepClassSpecificationVisitorFactory(true, true, true);

                    // Doesn't the specification match any program classes?
                    ClassCounter programClassCounter = new ClassCounter();

                    programClassPool.accept(
                        visitorFactory
                            .createClassPoolVisitor(keepClassSpecification,
                                                    programClassCounter,
                                                    null,
                                                    null,
                                                    null));

                    if (programClassCounter.getCount() == 0)
                    {
                        // Print out notes about any matched library classes.
                        libraryClassPool.accept(
                            visitorFactory
                                .createClassPoolVisitor(keepClassSpecification,
                                                        this,
                                                        null,
                                                        null,
                                                        null));
                    }
                }
            }
        }
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass) {}


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        String className = libraryClass.getName();
        notePrinter.print(className,
                          "Note: the configuration explicitly specifies '" +
                          ClassUtil.externalClassName(keepName) +
                          "' to keep library class '" +
                          ClassUtil.externalClassName(className) +
                          "'");
    }
}
