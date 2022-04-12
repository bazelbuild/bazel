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
package proguard.optimize.peephole;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;

import java.util.*;

/**
 * This ClassVisitor disambiguates line numbers, in the classes that it
 * visits. It shifts line numbers that originate from different classes
 * (e.g. due to method inlining or class merging) to blocks that don't
 * overlap with the main line numbers and with each other. The line numbers
 * then uniquely identify the inlined and merged code in the classes.
 *
 * @author Eric Lafortune
 */
public class LineNumberLinearizer
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor,
             AttributeVisitor,
             LineNumberInfoVisitor
{
    private static final boolean DEBUG = false;

    public  static final int SHIFT_ROUNDING       = 1000;
    private static final int SHIFT_ROUNDING_LIMIT = 50000;


    private Stack          enclosingLineNumbers = new Stack();
    private LineNumberInfo previousLineNumberInfo;
    private int            highestUsedLineNumber;
    private int            currentLineNumberShift;


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Find the highest line number in the entire class.
        LineNumberRangeFinder lineNumberRangeFinder =
            new LineNumberRangeFinder();

        programClass.methodsAccept(new AllAttributeVisitor(true,
                                   new AllLineNumberInfoVisitor(
                                   lineNumberRangeFinder)));

        // Are there any inlined line numbers?
        if (lineNumberRangeFinder.hasSource())
        {
            // Remember the minimum initial shift.
            highestUsedLineNumber = lineNumberRangeFinder.getHighestLineNumber();

            // Shift the inlined line numbers.
            programClass.methodsAccept(this);
        }
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        programMethod.attributesAccept(programClass, this);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        codeAttribute.attributesAccept(clazz, method, this);
    }


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        if (DEBUG)
        {
            System.out.println("LineNumberLinearizer ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]:");
        }

        enclosingLineNumbers.clear();
        previousLineNumberInfo = null;

        // Process all line numbers.
        lineNumberTableAttribute.lineNumbersAccept(clazz, method, codeAttribute, this);
    }


    // Implementations for LineNumberInfoVisitor.

    public void visitLineNumberInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberInfo lineNumberInfo)
    {
        String source = lineNumberInfo.getSource();

        if (DEBUG)
        {
            System.out.print("    [" + lineNumberInfo.u2startPC + "] line " + lineNumberInfo.u2lineNumber + (source == null ? "" : " [" + source + "]"));
        }

        // Is it an inlined line number?
        if (source != null)
        {
            ExtendedLineNumberInfo extendedLineNumberInfo =
                (ExtendedLineNumberInfo)lineNumberInfo;

            int lineNumber = extendedLineNumberInfo.u2lineNumber;

            // Are we entering or exiting a new inlined block?
            if (previousLineNumberInfo == null ||
                previousLineNumberInfo.getSource() != source)
            {
                // Are we entering a new inlined block?
                if (lineNumber != MethodInliner.INLINED_METHOD_END_LINE_NUMBER)
                {
                    // Remember information about the inlined block.
                    enclosingLineNumbers.push(previousLineNumberInfo != null ?
                        new MyLineNumberBlock(currentLineNumberShift,
                                              previousLineNumberInfo.u2lineNumber,
                                              previousLineNumberInfo.getSource()) :
                        new MyLineNumberBlock(0, 0, null));

                    // Parse the end line number from the source string,
                    // so we know how large a block this will be.
                    int separatorIndex1 = source.indexOf(':');
                    int separatorIndex2 = source.indexOf(':', separatorIndex1 + 1);

                    int startLineNumber = Integer.parseInt(source.substring(separatorIndex1 + 1, separatorIndex2));
                    int endLineNumber   = Integer.parseInt(source.substring(separatorIndex2 + 1));

                    // Start shifting, if necessary, so the block ends up beyond
                    // the highest used line number. We're striving for rounded
                    // shifts, unless we've reached a given limit, to avoid
                    // running out of line numbers too quickly.
                    currentLineNumberShift =
                        highestUsedLineNumber > SHIFT_ROUNDING_LIMIT ?
                            highestUsedLineNumber - startLineNumber + 1 :
                        startLineNumber > highestUsedLineNumber ? 0 :
                            (highestUsedLineNumber - startLineNumber + SHIFT_ROUNDING)
                            / SHIFT_ROUNDING * SHIFT_ROUNDING;

                    highestUsedLineNumber = endLineNumber + currentLineNumberShift;

                    if (DEBUG)
                    {
                        System.out.print(" (enter with shift "+currentLineNumberShift+")");
                    }

                    // Apply the shift.
                    lineNumberInfo.u2lineNumber += currentLineNumberShift;
                }

                // TODO: There appear to be cases where the stack is empty at this point, so we've added a check.
                else if (enclosingLineNumbers.isEmpty())
                {
                    if (DEBUG)
                    {
                        System.err.println("Problem linearizing line numbers for optimized code ("+clazz.getName()+"."+method.getName(clazz)+")");
                    }
                }

                // Are we exiting an inlined block?
                else
                {
                    // Pop information about the enclosing line number.
                    MyLineNumberBlock lineNumberBlock =
                        (MyLineNumberBlock)enclosingLineNumbers.pop();

                    // Set this end of the block to the line at which it was
                    // inlined.
                    extendedLineNumberInfo.u2lineNumber = lineNumberBlock.enclosingLineNumber;
                    extendedLineNumberInfo.source       = lineNumberBlock.enclosingSource;

                    // Reset the shift to the shift of the block.
                    currentLineNumberShift = lineNumberBlock.lineNumberShift;

                    if (DEBUG)
                    {
                        System.out.print(" (exit to shift "+currentLineNumberShift+")");
                    }
                }
            }
            else
            {
                if (DEBUG)
                {
                    System.out.print(" (apply shift "+currentLineNumberShift+")");
                }

                // Apply the shift.
                lineNumberInfo.u2lineNumber += currentLineNumberShift;
            }
        }

        previousLineNumberInfo = lineNumberInfo;

        if (DEBUG)
        {
            System.out.println(" -> line " + lineNumberInfo.u2lineNumber);
        }
    }


    /**
     * This class represents a block of line numbers that originates from the
     * same inlined method.
     */
    private static class MyLineNumberBlock
    {
        public final int    lineNumberShift;
        public final int    enclosingLineNumber;
        public final String enclosingSource;

        public MyLineNumberBlock(int    lineNumberShift,
                                 int    enclosingLineNumber,
                                 String enclosingSource)
        {
            this.lineNumberShift     = lineNumberShift;
            this.enclosingLineNumber = enclosingLineNumber;
            this.enclosingSource     = enclosingSource;
        }
    }
}
