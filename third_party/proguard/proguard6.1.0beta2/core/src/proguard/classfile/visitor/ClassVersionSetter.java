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
package proguard.classfile.visitor;

import proguard.classfile.*;

import java.util.Set;

/**
 * This <code>ClassVisitor</code> sets the version number of the program classes
 * that it visits.
 *
 * @author Eric Lafortune
 */
public class ClassVersionSetter implements ClassVisitor
{
    private final int classVersion;

    private final Set newerClassVersions;


    /**
     * Creates a new ClassVersionSetter.
     * @param classVersion the class version number.
     */
    public ClassVersionSetter(int classVersion)
    {
        this(classVersion, null);
    }


    /**
     * Creates a new ClassVersionSetter that also stores any newer class version
     * numbers that it encounters while visiting program classes.
     * @param classVersion       the class version number.
     * @param newerClassVersions the <code>Set</code> in which newer class
     *                           version numbers can be collected.
     */
    public ClassVersionSetter(int classVersion,
                              Set newerClassVersions)
    {
        this.classVersion       = classVersion;
        this.newerClassVersions = newerClassVersions;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (programClass.u4version > classVersion &&
            newerClassVersions != null)
        {
            newerClassVersions.add(new Integer(programClass.u4version));
        }

        programClass.u4version = classVersion;
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Library classes don't have version numbers.
    }
}
