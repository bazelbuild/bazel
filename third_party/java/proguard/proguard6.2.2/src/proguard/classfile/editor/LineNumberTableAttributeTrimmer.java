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

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.SimplifiedVisitor;

import java.util.Arrays;

/**
 * This AttributeVisitor trims the line number table attributes that it visits.
 *
 * @author Eric Lafortune
 */
public class LineNumberTableAttributeTrimmer
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        LineNumberInfo[] lineNumberTable       = lineNumberTableAttribute.lineNumberTable;
        int              lineNumberTableLength = lineNumberTableAttribute.u2lineNumberTableLength;

        // Overwrite all empty line number entries.
        int newIndex = 0;
        for (int index = 0; index < lineNumberTableLength; index++)
        {
            LineNumberInfo lineNumberInfo = lineNumberTable[index];

            int startPC    = lineNumberInfo.u2startPC;
            int lineNumber = lineNumberInfo.u2lineNumber;

            // The offset must lie inside the code.
            // The offset must be smaller than the next one.
            // The line number should be different from the previous one.
            if (startPC < codeAttribute.u4codeLength             &&

                (index == lineNumberTableLength - 1 ||
                 startPC < lineNumberTable[index + 1].u2startPC) &&

                (index == 0 ||
                 lineNumber != lineNumberTable[index - 1].u2lineNumber))
            {
                lineNumberTable[newIndex++] = lineNumberInfo;
            }
        }

        // Clear the unused array entries.
        Arrays.fill(lineNumberTable, newIndex, lineNumberTableAttribute.u2lineNumberTableLength, null);

        lineNumberTableAttribute.u2lineNumberTableLength = newIndex;
    }
}
