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
package proguard.classfile.visitor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.ExceptionInfoVisitor;
import proguard.classfile.constant.visitor.ConstantVisitor;

/**
 * This <code>ExceptionInfoVisitor</code> lets a given
 * <code>ConstantVisitor</code> visit all catch class constants of exceptions
 * that it visits.
 *
 * @author Eric Lafortune
 */
public class ExceptionHandlerConstantVisitor
implements   ExceptionInfoVisitor
{
    private final ConstantVisitor constantVisitor;


    /**
     * Creates a new ExceptionHandlerConstantVisitor.
     * @param constantVisitor the ConstantVisitor that will visit the catch
     *                        class constants.
     */
    public ExceptionHandlerConstantVisitor(ConstantVisitor constantVisitor)
    {
        this.constantVisitor = constantVisitor;
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        int catchType = exceptionInfo.u2catchType;
        if (catchType != 0)
        {
            clazz.constantPoolEntryAccept(catchType, constantVisitor);
        }
    }
}