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
package proguard.obfuscate;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.optimize.peephole.LineNumberLinearizer;

import java.io.*;
import java.util.Stack;


/**
 * This ClassVisitor prints out the renamed classes and class members with
 * their old names and new names.
 *
 * @see ClassRenamer
 *
 * @author Eric Lafortune
 */
public class MappingPrinter
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             AttributeVisitor
{
    private final PrintWriter pw;

    // A field serving as a return value for the visitor methods.
    private boolean printed;


    /**
     * Creates a new MappingPrinter that prints to the given writer.
     * @param printWriter the writer to which to print.
     */
    public MappingPrinter(PrintWriter printWriter)
    {
        this.pw = printWriter;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        String name    = programClass.getName();
        String newName = ClassObfuscator.newClassName(programClass);

        // Print out the class mapping.
        pw.println(ClassUtil.externalClassName(name) +
                   " -> " +
                   ClassUtil.externalClassName(newName) +
                   ":");

        // Print out the class members.
        programClass.fieldsAccept(this);
        programClass.methodsAccept(this);
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        String fieldName           = programField.getName(programClass);
        String obfuscatedFieldName = MemberObfuscator.newMemberName(programField);
        if (obfuscatedFieldName == null)
        {
            obfuscatedFieldName = fieldName;
        }

        // Print out the field mapping.
        pw.println("    " +
                   ClassUtil.externalType(programField.getDescriptor(programClass)) + " " +
                   fieldName +
                   " -> " +
                   obfuscatedFieldName);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        String methodName           = programMethod.getName(programClass);
        String obfuscatedMethodName = MemberObfuscator.newMemberName(programMethod);
        if (obfuscatedMethodName == null)
        {
            obfuscatedMethodName = methodName;
        }

        // Print out the method mapping, if it has line numbers.
        printed = false;
        programMethod.attributesAccept(programClass, this);

        // Otherwise print out the method mapping without line numbers.
        if (!printed)
        {
            pw.println("    " +
                       ClassUtil.externalMethodReturnType(programMethod.getDescriptor(programClass)) + " " +
                       methodName                                                                    + JavaConstants.METHOD_ARGUMENTS_OPEN  +
                       ClassUtil.externalMethodArguments(programMethod.getDescriptor(programClass))  + JavaConstants.METHOD_ARGUMENTS_CLOSE +
                       " -> " +
                       obfuscatedMethodName);
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
        LineNumberInfo[] lineNumberTable       = lineNumberTableAttribute.lineNumberTable;
        int              lineNumberTableLength = lineNumberTableAttribute.u2lineNumberTableLength;

        String methodName           = method.getName(clazz);
        String methodDescriptor     = method.getDescriptor(clazz);
        String obfuscatedMethodName = MemberObfuscator.newMemberName(method);
        if (obfuscatedMethodName == null)
        {
            obfuscatedMethodName = methodName;
        }

        int lowestLineNumber  = lineNumberTableAttribute.getLowestLineNumber();
        int highestLineNumber = lineNumberTableAttribute.getHighestLineNumber();

        // Does the method have any local line numbers at all?
        if (lineNumberTableAttribute.getSource(codeAttribute.u4codeLength)  == null)
        {
            if (lowestLineNumber > 0)
            {
                // Print out the line number range of the method,
                // ignoring line numbers of any inlined methods.
                pw.println("    " +
                           lowestLineNumber                                                + ":" +
                           highestLineNumber                                               + ":" +
                           ClassUtil.externalMethodReturnType(method.getDescriptor(clazz)) + " " +
                           methodName                                                      + JavaConstants.METHOD_ARGUMENTS_OPEN  +
                           ClassUtil.externalMethodArguments(method.getDescriptor(clazz))  + JavaConstants.METHOD_ARGUMENTS_CLOSE +
                           " -> " +
                           obfuscatedMethodName);
            }
            else
            {
                // Print out the method mapping without line numbers.
                pw.println("    " +
                           ClassUtil.externalMethodReturnType(method.getDescriptor(clazz)) + " " +
                           methodName                                                      + JavaConstants.METHOD_ARGUMENTS_OPEN  +
                           ClassUtil.externalMethodArguments(method.getDescriptor(clazz))  + JavaConstants.METHOD_ARGUMENTS_CLOSE +
                           " -> " +
                           obfuscatedMethodName);
            }
        }

        // Print out the line numbers of any inlined methods and their
        // enclosing methods.
        Stack enclosingLineNumbers = new Stack();

        LineNumberInfo previousInfo = new LineNumberInfo(0, 0);

        for (int index = 0; index < lineNumberTableLength; index++)
        {
            LineNumberInfo info = lineNumberTable[index];

            // Are we entering or exiting an inlined block (or a merged block)?
            // We're testing on the identities out of convenience.
            String previousSource = previousInfo.getSource();
            String source         = info.getSource();
            // Source can be null for injected code.
            if (source != null && source != previousSource)
            {
                // Are we entering or exiting the block?
                int previousLineNumber = previousInfo.u2lineNumber;
                int lineNumber         = info.u2lineNumber;
                if (lineNumber > previousLineNumber)
                {
                    // We're entering an inlined block.
                    // Accumulate its enclosing line numbers, so they can be
                    // printed out for each inlined block.
                    if (index > 0)
                    {
                        enclosingLineNumbers.push(previousInfo);
                    }

                    printInlinedMethodMapping(clazz.getName(),
                                              methodName,
                                              methodDescriptor,
                                              info,
                                              enclosingLineNumbers,
                                              obfuscatedMethodName);
                }
                // TODO: There appear to be cases where the stack is empty at this point, so we've added a check.
                else if (!enclosingLineNumbers.isEmpty())
                {
                    // We're exiting an inlined block.
                    // Pop its enclosing line number.
                    enclosingLineNumbers.pop();
                }
            }
            else if (source == null && previousSource != null)
            {
                // TODO: There appear to be cases where the stack is empty at this point, so we've added a check.
                if (!enclosingLineNumbers.isEmpty())
                {
                    // When exiting a top-level inlined block, the source might be null.
                    // See LineNumberLinearizer, line 185, exiting an inlined block.
                    enclosingLineNumbers.pop();
                }
            }

            previousInfo = info;
        }

        printed = true;
    }


    // Small utility methods.

    /**
     * Prints out the mapping of the specified inlined methods and its
     * enclosing methods.
     */
    private void printInlinedMethodMapping(String         className,
                                           String         methodName,
                                           String         methodDescriptor,
                                           LineNumberInfo inlinedInfo,
                                           Stack          enclosingLineNumbers,
                                           String         obfuscatedMethodName)
    {
        String source = inlinedInfo.getSource();

        // Parse the information from the source string of the
        // inlined method.
        int separatorIndex1 = source.indexOf('.');
        int separatorIndex2 = source.indexOf('(', separatorIndex1 + 1);
        int separatorIndex3 = source.indexOf(':', separatorIndex2 + 1);
        int separatorIndex4 = source.indexOf(':', separatorIndex3 + 1);

        String inlinedClassName        = source.substring(0, separatorIndex1);
        String inlinedMethodName       = source.substring(separatorIndex1 + 1, separatorIndex2);
        String inlinedMethodDescriptor = source.substring(separatorIndex2, separatorIndex3);
        String inlinedRange            = source.substring(separatorIndex3);

        int startLineNumber = Integer.parseInt(source.substring(separatorIndex3 + 1, separatorIndex4));
        int endLineNumber   = Integer.parseInt(source.substring(separatorIndex4 + 1));

        // Compute the shifted line number range.
        int shiftedStartLineNumber = inlinedInfo.u2lineNumber;
        int shiftedEndLineNumber   = shiftedStartLineNumber + endLineNumber - startLineNumber;

        // Print out the line number range of the inlined method.
        pw.println("    " +
                   shiftedStartLineNumber                                      + ":" +
                   shiftedEndLineNumber                                        + ":" +
                   ClassUtil.externalMethodReturnType(inlinedMethodDescriptor) + " " +
                   (inlinedClassName.equals(className) ? "" :
                   ClassUtil.externalClassName(inlinedClassName)               + JavaConstants.PACKAGE_SEPARATOR)     +
                   inlinedMethodName                                           + JavaConstants.METHOD_ARGUMENTS_OPEN  +
                   ClassUtil.externalMethodArguments(inlinedMethodDescriptor)  + JavaConstants.METHOD_ARGUMENTS_CLOSE +
                   inlinedRange                                                + " -> " +
                   obfuscatedMethodName);

        // Print out the line numbers of the accumulated enclosing
        // methods.
        for (int enclosingIndex = enclosingLineNumbers.size()-1; enclosingIndex >= 0; enclosingIndex--)
        {
            LineNumberInfo enclosingInfo =
                (LineNumberInfo)enclosingLineNumbers.get(enclosingIndex);

            printEnclosingMethodMapping(className,
                                        methodName,
                                        methodDescriptor,
                                        shiftedStartLineNumber + ":" +
                                        shiftedEndLineNumber,
                                        enclosingInfo,
                                        obfuscatedMethodName);
        }
    }


    /**
     * Prints out the mapping of the specified enclosing method.
     */
    private void printEnclosingMethodMapping(String         className,
                                             String         methodName,
                                             String         methodDescriptor,
                                             String         shiftedRange,
                                             LineNumberInfo enclosingInfo,
                                             String         obfuscatedMethodName)
    {
        // Parse the information from the source string of the enclosing
        // method.
        String enclosingSource = enclosingInfo.getSource();

        String enclosingClassName;
        String enclosingMethodName;
        String enclosingMethodDescriptor;
        int    enclosingLineNumber;

        if (enclosingSource == null)
        {
            enclosingClassName        = className;
            enclosingMethodName       = methodName;
            enclosingMethodDescriptor = methodDescriptor;
            enclosingLineNumber       = enclosingInfo.u2lineNumber;
        }
        else
        {
            int enclosingSeparatorIndex1 = enclosingSource.indexOf('.');
            int enclosingSeparatorIndex2 = enclosingSource.indexOf('(', enclosingSeparatorIndex1 + 1);
            int enclosingSeparatorIndex3 = enclosingSource.indexOf(':', enclosingSeparatorIndex2 + 1);
            int enclosingSeparatorIndex4 = enclosingSource.indexOf(':', enclosingSeparatorIndex3 + 1);

            // We need the first line number to correct the shifted enclosing
            // line number back to its original range.
            int firstLineNumber = Integer.parseInt(enclosingSource.substring(enclosingSeparatorIndex3 + 1, enclosingSeparatorIndex4));

            enclosingClassName        = enclosingSource.substring(0, enclosingSeparatorIndex1);
            enclosingMethodName       = enclosingSource.substring(enclosingSeparatorIndex1 + 1, enclosingSeparatorIndex2);
            enclosingMethodDescriptor = enclosingSource.substring(enclosingSeparatorIndex2, enclosingSeparatorIndex3);
            enclosingLineNumber       = (enclosingInfo.u2lineNumber - firstLineNumber) % LineNumberLinearizer.SHIFT_ROUNDING + firstLineNumber;
        }

        // Print out the line number of the enclosing method.
        pw.println("    " +
                   shiftedRange                                                  + ":" +
                   ClassUtil.externalMethodReturnType(enclosingMethodDescriptor) + " " +
                   (enclosingClassName.equals(className) ? "" :
                   ClassUtil.externalClassName(enclosingClassName)               + JavaConstants.PACKAGE_SEPARATOR)     +
                   enclosingMethodName                                           + JavaConstants.METHOD_ARGUMENTS_OPEN  +
                   ClassUtil.externalMethodArguments(enclosingMethodDescriptor)  + JavaConstants.METHOD_ARGUMENTS_CLOSE + ":" +
                   enclosingLineNumber                                           + " -> " +
                   obfuscatedMethodName);
    }
}
