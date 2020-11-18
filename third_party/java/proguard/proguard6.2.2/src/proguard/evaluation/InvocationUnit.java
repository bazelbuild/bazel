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
package proguard.evaluation;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.instruction.ConstantInstruction;
import proguard.evaluation.value.Value;

/**
 * This interface sets up the variables for entering a method,
 * and it updates the stack for the invocation of a class member.
 *
 * @author Eric Lafortune
 */
public interface InvocationUnit
{
    /**
     * Sets up the given variables for entering the given method.
     */
    public void enterMethod(Clazz     clazz,
                            Method    method,
                            Variables variables);


    /**
     * Exits the given method with the given return value.
     */
    public void exitMethod(Clazz  clazz,
                           Method method,
                           Value  returnValue);


    /**
     * Sets up the given stack for entering the given exception handler.
     */
    public void enterExceptionHandler(Clazz         clazz,
                                      Method        method,
                                      CodeAttribute codeAttribute,
                                      int           offset,
                                      int           catchType,
                                      Stack         stack);


    /**
     * Updates the given stack corresponding to the execution of the given
     * field or method reference instruction.
     */
    public void invokeMember(Clazz               clazz,
                             Method              method,
                             CodeAttribute       codeAttribute,
                             int                 offset,
                             ConstantInstruction constantInstruction,
                             Stack               stack);
}
