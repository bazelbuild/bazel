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
import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.attribute.visitor.AllAttributeVisitor;
import proguard.classfile.visitor.*;

/**
 * This class visitor removes Gson annotations that are not required anymore
 * after the Gson optimizations are applied.
 *
 * @author Rob Coekaerts
 * @author Lars Vandenbergh
 */
public class GsonAnnotationCleaner
implements   ClassVisitor
{

    private final GsonRuntimeSettings gsonRuntimeSettings;


    /**
     * Creates a new GsonAnnotationCleaner.
     *
     * @param gsonRuntimeSettings keeps track of all GsonBuilder invocations.
     */
    public GsonAnnotationCleaner(GsonRuntimeSettings gsonRuntimeSettings)
    {
        this.gsonRuntimeSettings = gsonRuntimeSettings;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        final Object mark = new Object();

        // Mark annotations when we are sure that they are not required
        // anymore by GSON.
        if (!gsonRuntimeSettings.setFieldNamingPolicy &&
            !gsonRuntimeSettings.setFieldNamingStrategy)
        {
            programClass.fieldsAccept(
                new AllAttributeVisitor(
                new AllAnnotationVisitor(
                new AnnotationTypeFilter(GsonClassConstants.ANNOTATION_TYPE_SERIALIZED_NAME,
                new VisitorInfoSetter(mark)))));
        }

        programClass.fieldsAccept(
            new AllAttributeVisitor(
            new AllAnnotationVisitor(
            new AnnotationTypeFilter(GsonClassConstants.ANNOTATION_TYPE_EXPOSE,
            new VisitorInfoSetter(mark)))));

        // Remove marked annotations.
        programClass.fieldsAccept(
            new AllAttributeVisitor(
            new MarkedAnnotationDeleter(mark)));

        // Unmark all annotations on fields.
        programClass.fieldsAccept(
            new AllAttributeVisitor(
            new AllAnnotationVisitor(
            new VisitorInfoSetter(null))));
    }


    public void visitLibraryClass(LibraryClass libraryClass) {}
}
