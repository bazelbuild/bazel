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
import proguard.classfile.attribute.visitor.ExceptionInfoVisitor;

/**
 * This ExceptionInfoVisitor adds all exception information that it visits to
 * the given target code attribute.
 *
 * @author Eric Lafortune
 */
public class ExceptionInfoAdder
implements   ExceptionInfoVisitor
{
    private final ConstantAdder         constantAdder;
    private final CodeAttributeComposer codeAttributeComposer;


    /**
     * Creates a new ExceptionAdder that will copy exceptions into the given
     * target code attribute.
     */
    public ExceptionInfoAdder(ProgramClass          targetClass,
                              CodeAttributeComposer targetComposer)
    {
        constantAdder         = new ConstantAdder(targetClass);
        codeAttributeComposer = targetComposer;
    }


    // Implementations for ExceptionInfoVisitor.

    public void visitExceptionInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, ExceptionInfo exceptionInfo)
    {
        // Create a copy of the exception info.
        ExceptionInfo newExceptionInfo =
            new ExceptionInfo(exceptionInfo.u2startPC,
                              exceptionInfo.u2endPC,
                              exceptionInfo.u2handlerPC,
                              exceptionInfo.u2catchType == 0 ? 0 :
                                  constantAdder.addConstant(clazz, exceptionInfo.u2catchType));

        // Add the completed exception info.
        codeAttributeComposer.appendException(newExceptionInfo);
    }
}