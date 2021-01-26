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
import proguard.classfile.instruction.*;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.optimize.info.ExceptionInstructionChecker;

/**
 * This AttributeVisitor removes exception handlers that are unreachable in the
 * code attributes that it visits.
 *
 * @author Eric Lafortune
 */
public class UnreachableExceptionRemover
extends      SimplifiedVisitor
implements   AttributeVisitor,
             ExceptionInfoVisitor
{
    private final ExceptionInfoVisitor extraExceptionInfoVisitor;


    /**
     * Creates a new UnreachableExceptionRemover.
     */
    public UnreachableExceptionRemover()
    {
        this(null);
    }


    /**
     * Creates a new UnreachableExceptionRemover.
     * @param extraExceptionInfoVisitor an optional extra visitor for all
     *                                  removed exceptions.
     */
    public UnreachableExceptionRemover(ExceptionInfoVisitor extraExceptionInfoVisitor)
    {
        this.extraExceptionInfoVisitor = extraExceptionInfoVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Go over the exception table.
        codeAttribute.exceptionsAccept(clazz, method, this);

        // Remove exceptions with empty code blocks.
        codeAttribute.u2exceptionTableLength =
            removeEmptyExceptions(codeAttribute.exceptionTable,
                                  codeAttribute.u2exceptionTableLength);
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        if (!mayThrowExceptions(clazz,
                                method,
                                codeAttribute,
                                exceptionInfo.u2startPC,
                                exceptionInfo.u2endPC))
        {
            // Make the code block empty.
            exceptionInfo.u2endPC = exceptionInfo.u2startPC;

            if (extraExceptionInfoVisitor != null)
            {
                extraExceptionInfoVisitor.visitExceptionInfo(clazz, method, codeAttribute, exceptionInfo);
            }
        }
    }


    // Small utility methods.

    /**
     * Returns whether the specified block of code may throw exceptions.
     */
    private boolean mayThrowExceptions(Clazz         clazz,
                                       Method        method,
                                       CodeAttribute codeAttribute,
                                       int           startOffset,
                                       int           endOffset)
    {
        byte[] code = codeAttribute.code;

        // Go over all instructions.
        int offset = startOffset;
        while (offset < endOffset)
        {
            // Get the current instruction.
            Instruction instruction = InstructionFactory.create(code, offset);

            // Check if it may be throwing exceptions.
            if (instruction.mayThrowExceptions())
            {
                return true;
            }

            // Go to the next instruction.
            offset += instruction.length(offset);
        }

        return false;
    }


    /**
     * Returns the given list of exceptions, without the ones that have empty
     * code blocks.
     */
    private int removeEmptyExceptions(ExceptionInfo[] exceptionInfos,
                                      int             exceptionInfoCount)
    {
        // Overwrite all empty exceptions.
        int newIndex = 0;
        for (int index = 0; index < exceptionInfoCount; index++)
        {
            ExceptionInfo exceptionInfo = exceptionInfos[index];
            if (exceptionInfo.u2startPC < exceptionInfo.u2endPC)
            {
                exceptionInfos[newIndex++] = exceptionInfo;
            }
        }

        return newIndex;
    }
}
