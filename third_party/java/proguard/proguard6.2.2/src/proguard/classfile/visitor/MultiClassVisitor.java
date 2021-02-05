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
import proguard.util.ArrayUtil;


/**
 * This ClassVisitor delegates all visits to each ClassVisitor
 * in a given list.
 *
 * @author Eric Lafortune
 */
public class MultiClassVisitor implements ClassVisitor
{
    private ClassVisitor[] classVisitors;
    private int            classVisitorCount;


    public MultiClassVisitor()
    {
        this.classVisitors = new ClassVisitor[16];
    }


    public MultiClassVisitor(ClassVisitor... classVisitors)
    {
        this.classVisitors     = classVisitors;
        this.classVisitorCount = classVisitors.length;
    }


    public void addClassVisitor(ClassVisitor classVisitor)
    {
        classVisitors =
            ArrayUtil.add(classVisitors,
                          classVisitorCount++,
                          classVisitor);
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        for (int index = 0; index < classVisitorCount; index++)
        {
            classVisitors[index].visitProgramClass(programClass);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        for (int index = 0; index < classVisitorCount; index++)
        {
            classVisitors[index].visitLibraryClass(libraryClass);
        }
    }
}
