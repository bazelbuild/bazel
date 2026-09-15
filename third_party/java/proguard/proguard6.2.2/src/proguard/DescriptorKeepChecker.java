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
import proguard.optimize.*;

import java.util.List;

/**
 * This class checks whether classes referenced by class members that are
 * marked to be kept are marked to be kept too.
 *
 * @author Eric Lafortune
 */
public class DescriptorKeepChecker
extends      SimplifiedVisitor
implements   MemberVisitor,
             ClassVisitor
{
    private final ClassPool      programClassPool;
    private final ClassPool      libraryClassPool;
    private final WarningPrinter notePrinter;

    // Some fields acting as parameters for the class visitor.
    private Clazz   referencingClass;
    private Member  referencingMember;
    private boolean isField;


    /**
     * Creates a new DescriptorKeepChecker.
     */
    public DescriptorKeepChecker(ClassPool      programClassPool,
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
        // Clean up any old visitor info.
        programClassPool.classesAccept(new ClassCleaner());
        libraryClassPool.classesAccept(new ClassCleaner());

        // Create a visitor for marking the seeds.
        KeepMarker keepMarker = new KeepMarker();
        ClassPoolVisitor classPoolvisitor =
            new KeepClassSpecificationVisitorFactory(true, true, true)
                .createClassPoolVisitor(keepSpecifications,
                                        keepMarker,
                                        keepMarker,
                                        keepMarker,
                                        null);
        // Mark the seeds.
        programClassPool.accept(classPoolvisitor);
        libraryClassPool.accept(classPoolvisitor);

        // Print out notes about argument types that are not being kept in
        // class members that are being kept.
        programClassPool.classesAccept(
            new AllMemberVisitor(
            new KeptMemberFilter(this)));
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        //referencingClass  = programClass;
        //referencingMember = programField;
        //isField           = true;
        //
        // Don't check the type, because it is not required for introspection.
        //programField.referencedClassesAccept(this);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        referencingClass  = programClass;
        referencingMember = programMethod;
        isField           = false;

        // Don't check the return type, because it is not required for
        // introspection (e.g. the return type of the special Enum methods).
        //programMethod.referencedClassesAccept(this);

        Clazz[] referencedClasses = programMethod.referencedClasses;
        if (referencedClasses != null)
        {
            int count = referencedClasses.length;

            // Adapt the count if the return type is a class type (not so
            // pretty; assuming test just checks for final ';').
            if (ClassUtil.isInternalClassType(programMethod.getDescriptor(programClass)))
            {
                count--;
            }

            for (int index = 0; index < count; index++)
            {
                if (referencedClasses[index] != null)
                {
                    referencedClasses[index].accept(this);
                }
            }
        }
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (!KeepMarker.isKept(programClass))
        {
            notePrinter.print(referencingClass.getName(),
                              programClass.getName(),
                              "Note: the configuration keeps the entry point '" +
                              ClassUtil.externalClassName(referencingClass.getName()) +
                              " { " +
                              (isField ?
                                   ClassUtil.externalFullFieldDescription(0,
                                                                          referencingMember.getName(referencingClass),
                                                                          referencingMember.getDescriptor(referencingClass)) :
                                   ClassUtil.externalFullMethodDescription(referencingClass.getName(),
                                                                           0,
                                                                           referencingMember.getName(referencingClass),
                                                                           referencingMember.getDescriptor(referencingClass))) +
                              "; }', but not the descriptor class '" +
                              ClassUtil.externalClassName(programClass.getName()) +
                              "'");
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass) {}
}
