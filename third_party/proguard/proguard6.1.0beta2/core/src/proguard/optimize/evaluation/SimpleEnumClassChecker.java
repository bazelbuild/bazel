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
package proguard.optimize.evaluation;

import proguard.classfile.*;
import proguard.classfile.visitor.*;
import proguard.optimize.OptimizationInfoClassFilter;
import proguard.optimize.info.SimpleEnumMarker;

/**
 * This ClassVisitor marks all program classes that it visits as simple enums,
 * if their methods qualify.
 *
 * @author Eric Lafortune
 */
public class SimpleEnumClassChecker
implements   ClassVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("enum") != null;
    //*/


    private final ClassVisitor  simpleEnumMarker     = new OptimizationInfoClassFilter(
                                                       new SimpleEnumMarker(true));
    private final MemberVisitor virtualMemberChecker = new MemberAccessFilter(0,
                                                                              ClassConstants.ACC_PRIVATE |
                                                                              ClassConstants.ACC_STATIC,
                                                       new MemberToClassVisitor(
                                                       new SimpleEnumMarker(false)));


    // Implementations for ClassVisitor.

    public void visitLibraryClass(LibraryClass libraryClass) {}

    public void visitProgramClass(ProgramClass programClass)
    {
        // Does the class have the simple enum constructor?
        if (programClass.findMethod(ClassConstants.METHOD_NAME_INIT,
                                    ClassConstants.METHOD_TYPE_INIT_ENUM) != null)
        {
            if (DEBUG)
            {
                System.out.println("SimpleEnumClassChecker: ["+programClass.getName()+"] is a candidate simple enum, without extra fields");
            }

            // Mark it.
            simpleEnumMarker.visitProgramClass(programClass);

            // However, unmark it again if it has any non-private, non-static
            // fields or methods.
            programClass.fieldsAccept(virtualMemberChecker);
            programClass.methodsAccept(virtualMemberChecker);
        }
    }
}