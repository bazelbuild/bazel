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
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This AttributeVisitor adds the line numbers of all line number attributes
 * that it visits to the given target line number attribute. It ensures that
 * the sources of the line numbers are preserved or set.
 */
public class LineNumberInfoAdder
extends      SimplifiedVisitor
implements   AttributeVisitor,
             LineNumberInfoVisitor
{
    private final LineNumberTableAttributeEditor lineNumberTableAttributeEditor;

    private String source;


    /**
     * Creates a new LineNumberInfoAdder that will copy line numbers into the
     * given target line number table.
     */
    public LineNumberInfoAdder(LineNumberTableAttribute targetLineNumberTableAttribute)
    {
        this.lineNumberTableAttributeEditor = new LineNumberTableAttributeEditor(targetLineNumberTableAttribute);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitLineNumberTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberTableAttribute lineNumberTableAttribute)
    {
        // Remember the source.
        source =
            clazz.getName()                                + '.' +
            method.getName(clazz)                          +
            method.getDescriptor(clazz)                    + ':' +
            lineNumberTableAttribute.getLowestLineNumber() + ':' +
            lineNumberTableAttribute.getHighestLineNumber();

        // Copy all line numbers.
        lineNumberTableAttribute.lineNumbersAccept(clazz, method, codeAttribute, this);
    }


    // Implementations for LineNumberInfoVisitor.

    public void visitLineNumberInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LineNumberInfo lineNumberInfo)
    {
        // Make sure we have a source.
        String newSource = lineNumberInfo.getSource() != null ?
            lineNumberInfo.getSource() :
            source;

        // Create a new line number.
        LineNumberInfo newLineNumberInfo =
            new ExtendedLineNumberInfo(lineNumberInfo.u2startPC,
                                       lineNumberInfo.u2lineNumber,
                                       newSource);

        // Add it to the target.
        lineNumberTableAttributeEditor.addLineNumberInfo(newLineNumberInfo);
    }
}
