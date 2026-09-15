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
package proguard.shrink;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;

import java.io.PrintWriter;


/**
 * This ClassVisitor prints out the classes and class members that have been
 * marked as being used (or not used).
 *
 * @see UsageMarker
 *
 * @author Eric Lafortune
 */
public class UsagePrinter
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             AttributeVisitor
{
    private final UsageMarker usageMarker;
    private final boolean     printUnusedItems;
    private final PrintWriter pw;

    // A field to remember the class name, if a header is needed for class members.
    private String      className;


    /**
     * Creates a new UsagePrinter that prints to the given writer.
     * @param usageMarker      the usage marker that was used to mark the
     *                         classes and class members.
     * @param printUnusedItems a flag that indicates whether only unused items
     *                         should be printed, or alternatively, only used
     *                         items.
     * @param printWriter      the writer to which to print.
     */
    public UsagePrinter(UsageMarker usageMarker,
                        boolean     printUnusedItems,
                        PrintWriter printWriter)
    {
        this.usageMarker      = usageMarker;
        this.printUnusedItems = printUnusedItems;
        this.pw               = printWriter;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (usageMarker.isUsed(programClass))
        {
            if (printUnusedItems)
            {
                className = programClass.getName();

                programClass.fieldsAccept(this);
                programClass.methodsAccept(this);

                className = null;
            }
            else
            {
                pw.println(ClassUtil.externalClassName(programClass.getName()));
            }
        }
        else
        {
            if (printUnusedItems)
            {
                pw.println(ClassUtil.externalClassName(programClass.getName()));
            }
        }
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        if (usageMarker.isUsed(programField) ^ printUnusedItems)
        {
            printClassNameHeader();

            pw.println("    " +
                       ClassUtil.externalFullFieldDescription(
                           programField.getAccessFlags(),
                           programField.getName(programClass),
                           programField.getDescriptor(programClass)));
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if (usageMarker.isUsed(programMethod) ^ printUnusedItems)
        {
            printClassNameHeader();

            pw.print("    ");
            programMethod.attributesAccept(programClass, this);
            pw.println(ClassUtil.externalFullMethodDescription(
                           programClass.getName(),
                           programMethod.getAccessFlags(),
                           programMethod.getName(programClass),
                           programMethod.getDescriptor(programClass)));
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        codeAttribute.attributesAccept(clazz, method, this);
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        pw.print(lineNumberTableAttribute.getLowestLineNumber() + ":" +
                 lineNumberTableAttribute.getHighestLineNumber() + ":");
    }


    // Small utility methods.

    /**
     * Prints the class name field. The field is then cleared, so it is not
     * printed again.
     */
    private void printClassNameHeader()
    {
        if (className != null)
        {
            pw.println(ClassUtil.externalClassName(className) + ":");
            className = null;
        }
    }
}
