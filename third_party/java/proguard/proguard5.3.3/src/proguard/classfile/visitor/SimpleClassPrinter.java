/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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
import proguard.classfile.util.ClassUtil;

import java.io.PrintStream;


/**
 * This <code>ClassVisitor</code> and <code>MemberVisitor</code>
 * prints out the class names of the classes it visits, and the full class
 * member descriptions of the class members it visits. The names are printed
 * in a readable, Java-like format. The access modifiers can be included or not.
 *
 * @author Eric Lafortune
 */
public class SimpleClassPrinter
implements   ClassVisitor,
             MemberVisitor
{
    private final boolean     printAccessModifiers;
    private final PrintStream ps;


    /**
     * Creates a new SimpleClassPrinter that prints to
     * <code>System.out</code>, including the access modifiers.
     */
    public SimpleClassPrinter()
    {
        this(true);
    }

    /**
     * Creates a new SimpleClassPrinter that prints to
     * <code>System.out</code>, with or without the access modifiers.
     */
    public SimpleClassPrinter(boolean printAccessModifiers)
    {
        this(printAccessModifiers, System.out);
    }

    /**
     * Creates a new SimpleClassPrinter that prints to the given
     * <code>PrintStream</code>, with or without the access modifiers.
     */
    public SimpleClassPrinter(boolean     printAccessModifiers,
                              PrintStream printStream)
    {
        this.printAccessModifiers = printAccessModifiers;
        this.ps                   = printStream;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        ps.println(ClassUtil.externalFullClassDescription(
                       printAccessModifiers ?
                           programClass.getAccessFlags() :
                           0,
                       programClass.getName()));
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        ps.println(ClassUtil.externalFullClassDescription(
                       printAccessModifiers ?
                           libraryClass.getAccessFlags() :
                           0,
                       libraryClass.getName()));
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        ps.println(ClassUtil.externalFullClassDescription(
                       printAccessModifiers ?
                           programClass.getAccessFlags() :
                           0,
                       programClass.getName()) +
                   ": " +
                   ClassUtil.externalFullFieldDescription(
                       printAccessModifiers ?
                           programField.getAccessFlags() :
                           0,
                       programField.getName(programClass),
                       programField.getDescriptor(programClass)));
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        ps.println(ClassUtil.externalFullClassDescription(
                       printAccessModifiers ?
                           programClass.getAccessFlags() :
                           0,
                       programClass.getName()) +
                   ": " +
                   ClassUtil.externalFullMethodDescription(
                       programClass.getName(),
                       printAccessModifiers ?
                           programMethod.getAccessFlags() :
                           0,
                       programMethod.getName(programClass),
                       programMethod.getDescriptor(programClass)));
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        ps.println(ClassUtil.externalFullClassDescription(
                       printAccessModifiers ?
                           libraryClass.getAccessFlags() :
                           0,
                       libraryClass.getName()) +
                   ": " +
                   ClassUtil.externalFullFieldDescription(
                       printAccessModifiers ?
                           libraryField.getAccessFlags() :
                           0,
                       libraryField.getName(libraryClass),
                       libraryField.getDescriptor(libraryClass)));
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        ps.println(ClassUtil.externalFullClassDescription(
                       printAccessModifiers ?
                           libraryClass.getAccessFlags() :
                           0,
                       libraryClass.getName()) +
                   ": " +
                   ClassUtil.externalFullMethodDescription(
                       libraryClass.getName(),
                       printAccessModifiers ?
                           libraryMethod.getAccessFlags() :
                           0,
                       libraryMethod.getName(libraryClass),
                       libraryMethod.getDescriptor(libraryClass)));
    }
}
