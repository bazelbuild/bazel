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
package proguard.optimize.gson;

import proguard.classfile.*;
import proguard.classfile.attribute.annotation.Annotation;
import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.attribute.visitor.AllAttributeVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;

import java.util.Arrays;

/**
 * This class visitor determines whether a given domain class can be optimized
 * by the GSON optimizations and traverses both the class and field hierarchy
 * to look for further domain classes.
 *
 * @author Lars Vandenbergh
 */
public class GsonDomainClassFinder
extends      SimplifiedVisitor
implements   ClassVisitor
{
    private static final boolean DEBUG = false;

    private final ClassPool                     typeAdapterClassPool;
    private final ClassPool                     gsonDomainClassPool;
    private final WarningPrinter                notePrinter;
    private final LocalOrAnonymousClassChecker  localOrAnonymousClassChecker =
        new LocalOrAnonymousClassChecker();
    private final TypeParameterClassChecker     typeParameterClassChecker    =
        new TypeParameterClassChecker();
    private final DuplicateJsonFieldNameChecker duplicateFieldNameChecker    =
        new DuplicateJsonFieldNameChecker();


    /**
     * Creates a new GsonDomainClassFinder.
     *
     * @param typeAdapterClassPool the class pool containing the classes for
     *                             which a custom Gson type adapter is
     *                             registered.
     * @param gsonDomainClassPool  the class pool to which the found domain
     *                             classes are added.
     * @param notePrinter          used to print notes about domain classes that
     *                             can not be handled by the Gson optimization.
     */
    public GsonDomainClassFinder(ClassPool      typeAdapterClassPool,
                                 ClassPool      gsonDomainClassPool,
                                 WarningPrinter notePrinter)
    {
        this.typeAdapterClassPool = typeAdapterClassPool;
        this.gsonDomainClassPool  = gsonDomainClassPool;
        this.notePrinter          = notePrinter;
    }


    // Implementations for ClassVisitor.

    @Override
    public void visitProgramClass(ProgramClass programClass)
    {
        if (gsonDomainClassPool.getClass(programClass.getName()) == null)
        {
            // Local or anonymous classes are excluded by GSON.
            programClass.accept(localOrAnonymousClassChecker);
            if (localOrAnonymousClassChecker.isLocalOrAnonymous())
            {
                // No need to note here because this is not handled
                // by GSON either.
                return;
            }

            if(librarySuperClassCount(programClass) != 0)
            {
                note(programClass.getName(),
                     "Note: " + ClassUtil.externalClassName(programClass.getName() +
                     " can not be optimized for GSON because" +
                     " it is or inherits from a library class."));
                return;
            }

            if(gsonSuperClassCount(programClass) != 0)
            {
                note(programClass.getName(),
                     "Note: " + ClassUtil.externalClassName(programClass.getName() +
                     " can not be optimized for GSON because" +
                     " it is or inherits from a GSON API class."));
                return;
            }

            // Classes with fields that have generic type parameters are
            // not supported by our optimization as it is rather complex
            // to derive all possible type arguments and generate methods
            // for each case.
            typeParameterClassChecker.hasFieldWithTypeParameter = false;
            programClass.hierarchyAccept(true,
                                         true,
                                         false,
                                         false,
                                         typeParameterClassChecker);
            if (typeParameterClassChecker.hasFieldWithTypeParameter)
            {
                note(programClass.getName(),
                     "Note: " + ClassUtil.externalClassName(programClass.getName() +
                     " can not be optimized for GSON because" +
                     " it uses generic type variables."));
                return;
            }

            // Class with duplicate field names are not supported by
            // GSON either.
            duplicateFieldNameChecker.hasDuplicateJsonFieldNames = false;
            programClass.hierarchyAccept(true,
                                         true,
                                         false,
                                         false,
                                         duplicateFieldNameChecker);
            if (duplicateFieldNameChecker.hasDuplicateJsonFieldNames)
            {
                note(programClass.getName(),
                     "Note: " + ClassUtil.externalClassName(programClass.getName() +
                     " can not be optimized for GSON because" +
                     " it contains duplicate field names in its JSON representation."));
                return;
            }

            // Classes for which type adapters were registered are not optimized.
            ClassCounter typeAdapterClassCounter = new ClassCounter();
            programClass.hierarchyAccept(true,
                                         true,
                                         false,
                                         false,
                                         new ClassPresenceFilter(typeAdapterClassPool,
                                             typeAdapterClassCounter, null));
            if (typeAdapterClassCounter.getCount() > 0)
            {
                note(programClass.getName(),
                     "Note: " + ClassUtil.externalClassName(programClass.getName() +
                     " can not be optimized for GSON because" +
                     " a custom type adapter is registered for it."));
                return;
            }

            // Classes that contain any JsonAdapter annotations are not optimized.
            AnnotationFinder annotationFinder = new AnnotationFinder();
            programClass.hierarchyAccept(true,
                                         true,
                                         false,
                                         false,
                                         new MultiClassVisitor(
                                             new AllAttributeVisitor(true,
                                             new AllAnnotationVisitor(
                                             new AnnotationTypeFilter(GsonClassConstants.ANNOTATION_TYPE_JSON_ADAPTER,
                                             annotationFinder)))));
            if (annotationFinder.found)
            {
                note(programClass.getName(),
                     "Note: " + ClassUtil.externalClassName(programClass.getName() +
                     " can not be optimized for GSON because" +
                     " it contains a JsonAdapter annotation."));
                return;
            }

            if ((programClass.getAccessFlags() & ClassConstants.ACC_INTERFACE) == 0)
            {
                if (DEBUG)
                {
                    System.out.println("GsonDomainClassFinder: adding domain class " +
                                       programClass.getName());
                }

                // Add type occurring in toJson() invocation to domain class pool.
                gsonDomainClassPool.addClass(programClass);

                // Recursively visit the fields of the domain class and consider
                // their classes as domain classes too.
                programClass.fieldsAccept(
                    new MemberAccessFilter(0, ClassConstants.ACC_SYNTHETIC,
                    new MultiMemberVisitor(
                        new MemberDescriptorReferencedClassVisitor(this),
                        new AllAttributeVisitor(
                        new SignatureAttributeReferencedClassVisitor(this)))));
            }

            // Consider super and sub classes as domain classes too.
            programClass.hierarchyAccept(false,
                                         true,
                                         false,
                                         true,
                                         this);
        }
    }

    @Override
    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Library classes can not be optimized.
    }


    // Utility methods.

    private int librarySuperClassCount(ProgramClass programClass)
    {
        ClassCounter nonObjectLibrarySuperClassCounter = new ClassCounter();
        programClass.hierarchyAccept(true,
                                     true,
                                     false,
                                     false,
                                     new LibraryClassFilter(
                                     new ClassNameFilter(Arrays.asList("!java/lang/Object", "!java/lang/Enum"),
                                     nonObjectLibrarySuperClassCounter)));
        return nonObjectLibrarySuperClassCounter.getCount();
    }

    private int gsonSuperClassCount(ProgramClass programClass)
    {
        ClassCounter gsonSuperClassCounter = new ClassCounter();
        programClass.hierarchyAccept(true,
                                     true,
                                     false,
                                     false,
                                     new ProgramClassFilter(
                                     new ClassNameFilter("com/google/gson/**",
                                     gsonSuperClassCounter)));
        return gsonSuperClassCounter.getCount();
    }


    private void note(String className, String note)
    {
        if (notePrinter != null)
        {
            notePrinter.print(className, note);
            notePrinter.print(className, "      You should consider keeping this class and its fields.");
        }
    }


    private class AnnotationFinder
    extends       SimplifiedVisitor
    implements    AnnotationVisitor
    {
        private boolean found;

        @Override
        public void visitAnnotation(Clazz clazz, Annotation annotation)
        {
            found = true;
        }
    }

}
