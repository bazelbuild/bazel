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
import proguard.classfile.visitor.ClassVisitor;

/**
 * This ClassVisitor writes out notes about the class files that it visits
 * being duplicates.
 *
 * @author Eric Lafortune
 */
public class DuplicateClassPrinter implements ClassVisitor
{
    private final WarningPrinter notePrinter;


    /**
     * Creates a new DuplicateClassVisitor.
     */
    public DuplicateClassPrinter(WarningPrinter notePrinter)
    {
        this.notePrinter = notePrinter;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        notePrinter.print(programClass.getName(),
                          "Note: duplicate definition of program class [" +
                          ClassUtil.externalClassName(programClass.getName()) + "]");
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        notePrinter.print(libraryClass.getName(),
                          "Note: duplicate definition of library class [" +
                          ClassUtil.externalClassName(libraryClass.getName()) + "]");
    }
}
