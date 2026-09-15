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

import proguard.classfile.attribute.ExceptionsAttribute;

/**
 * This class can add exceptions to a given exceptions attribute.
 * Exceptions to be added must have been added to the constant pool and filled
 * out beforehand.
 *
 * @author Eric Lafortune
 */
public class ExceptionsAttributeEditor
{
    private ExceptionsAttribute targetExceptionsAttribute;


    /**
     * Creates a new ExceptionsAttributeEditor that will edit exceptions in the
     * given exceptions attribute.
     */
    public ExceptionsAttributeEditor(ExceptionsAttribute targetExceptionsAttribute)
    {
        this.targetExceptionsAttribute = targetExceptionsAttribute;
    }


    /**
     * Adds a given exception to the exceptions attribute.
     */
    public void addException(int exceptionIndex)
    {
        int   exceptionIndexTableLength = targetExceptionsAttribute.u2exceptionIndexTableLength;
        int[] exceptionIndexTable       = targetExceptionsAttribute.u2exceptionIndexTable;

        // Make sure there is enough space for the new exception.
        if (exceptionIndexTable.length <= exceptionIndexTableLength)
        {
            targetExceptionsAttribute.u2exceptionIndexTable = new int[exceptionIndexTableLength+1];
            System.arraycopy(exceptionIndexTable, 0,
                             targetExceptionsAttribute.u2exceptionIndexTable, 0,
                             exceptionIndexTableLength);
            exceptionIndexTable = targetExceptionsAttribute.u2exceptionIndexTable;
        }

        // Add the exception.
        exceptionIndexTable[targetExceptionsAttribute.u2exceptionIndexTableLength++] = exceptionIndex;
    }
}
