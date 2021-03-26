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
import proguard.util.ArrayUtil;

/**
 * This class can add exceptions to the exception table of a given code
 * attribute. The exceptions must have been filled out beforehand.
 *
 * @author Eric Lafortune
 */
public class ExceptionInfoEditor
{
    private final CodeAttribute codeAttribute;


    /**
     * Creates a new ExceptionInfoEditor that can add exceptions to the
     * given code attribute.
     */
    public ExceptionInfoEditor(CodeAttribute codeAttribute)
    {
        this.codeAttribute = codeAttribute;
    }


    /**
     * Prepends the given exception to the exception table.
     */
    void prependException(ExceptionInfo exceptionInfo)
    {
        ExceptionInfo[] exceptionTable       = codeAttribute.exceptionTable;
        int             exceptionTableLength = codeAttribute.u2exceptionTableLength;

        int newExceptionTableLength = exceptionTableLength + 1;

        // Is the exception table large enough?
        if (exceptionTable.length < newExceptionTableLength)
        {
            ExceptionInfo[] newExceptionTable =
                new ExceptionInfo[newExceptionTableLength];

            System.arraycopy(exceptionTable, 0,
                             newExceptionTable, 1,
                             exceptionTableLength);
            newExceptionTable[0] = exceptionInfo;

            codeAttribute.exceptionTable = newExceptionTable;
        }
        else
        {
            System.arraycopy(exceptionTable, 0,
                             exceptionTable, 1,
                             exceptionTableLength);
            exceptionTable[0] = exceptionInfo;
        }

        codeAttribute.u2exceptionTableLength = newExceptionTableLength;
    }


    /**
     * Appends the given exception to the exception table.
     */
    void appendException(ExceptionInfo exceptionInfo)
    {
        codeAttribute.exceptionTable =
            ArrayUtil.add(codeAttribute.exceptionTable,
                          codeAttribute.u2exceptionTableLength++,
                          exceptionInfo);
    }
}