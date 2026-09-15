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
package proguard.classfile.editor;

import proguard.classfile.attribute.*;

/**
 * This class can add line numbers to a given line number table attribute.
 * Line numbers to be added must have been filled out beforehand.
 *
 * @author Eric Lafortune
 */
public class LineNumberTableAttributeEditor
{
    private LineNumberTableAttribute targetLineNumberTableAttribute;


    /**
     * Creates a new LineNumberTableAttributeEditor that will edit line numbers
     * in the given line number table attribute.
     */
    public LineNumberTableAttributeEditor(LineNumberTableAttribute targetLineNumberTableAttribute)
    {
        this.targetLineNumberTableAttribute = targetLineNumberTableAttribute;
    }


    /**
     * Adds a given line number to the line number table attribute.
     */
    public void addLineNumberInfo(LineNumberInfo lineNumberInfo)
    {
        int              lineNumberTableLength = targetLineNumberTableAttribute.u2lineNumberTableLength;
        LineNumberInfo[] lineNumberTable       = targetLineNumberTableAttribute.lineNumberTable;

        // Make sure there is enough space for the new lineNumberInfo.
        if (lineNumberTable.length <= lineNumberTableLength)
        {
            targetLineNumberTableAttribute.lineNumberTable = new LineNumberInfo[lineNumberTableLength+1];
            System.arraycopy(lineNumberTable, 0,
                             targetLineNumberTableAttribute.lineNumberTable, 0,
                             lineNumberTableLength);
            lineNumberTable = targetLineNumberTableAttribute.lineNumberTable;
        }

        // Add the lineNumberInfo.
        lineNumberTable[targetLineNumberTableAttribute.u2lineNumberTableLength++] = lineNumberInfo;
    }
}