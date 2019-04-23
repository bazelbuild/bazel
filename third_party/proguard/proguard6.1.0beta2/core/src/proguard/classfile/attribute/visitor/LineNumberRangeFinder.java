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
package proguard.classfile.attribute.visitor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;

/**
 * This LineNumberInfoVisitor remembers the lowest and the highest line
 * numbers that it finds in all the line numbers that it visits. It
 * ignores the sources of the line numbers.
 */
public class LineNumberRangeFinder
implements   LineNumberInfoVisitor
{
    private int     lowestLineNumber  = Integer.MAX_VALUE;
    private int     highestLineNumber = 0;
    private boolean hasSource;


    /**
     * Returns the lowest line number that has been visited so far.
     */
    public int getLowestLineNumber()
    {
        return lowestLineNumber;
    }


    /**
     * Returns the highest line number that has been visited so far.
     */
    public int getHighestLineNumber()
    {
        return highestLineNumber;
    }


    /**
     * Returns whether any of the visited line numbers has a non-null source.
     */
    public boolean hasSource()
    {
        return hasSource;
    }


    // Implementations for LineNumberInfoVisitor.

    public void visitLineNumberInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberInfo lineNumberInfo)
    {
        int lineNumber = lineNumberInfo.u2lineNumber;

        // Remember the lowest line number.
        if (lowestLineNumber > lineNumber)
        {
            lowestLineNumber = lineNumber;
        }

        // Remember the highest line number.
        if (highestLineNumber < lineNumber)
        {
            highestLineNumber = lineNumber;
        }

        if (lineNumberInfo.getSource() != null)
        {
            hasSource = true;
        }
    }
}
