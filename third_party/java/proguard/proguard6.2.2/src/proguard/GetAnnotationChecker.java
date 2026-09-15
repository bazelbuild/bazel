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
package proguard;

import proguard.classfile.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.*;

/**
 * This constant visitor checks whether visited method references try to
 * access annotations.
 *
 * @author Eric Lafortune
 */
public class GetAnnotationChecker
extends      SimplifiedVisitor
implements   ConstantVisitor
{
    private final WarningPrinter notePrinter;


    /**
     * Creates a new GetAnnotationChecker.
     */
    public GetAnnotationChecker(WarningPrinter notePrinter)
    {
        this.notePrinter = notePrinter;
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitMethodrefConstant(Clazz clazz, MethodrefConstant methodrefConstant)
    {
        String className = methodrefConstant.getClassName(clazz);

        if (className.equals(ClassConstants.NAME_JAVA_LANG_CLASS)         ||
            className.equals(ClassConstants.NAME_JAVA_LANG_REFLECT_FIELD) ||
            className.equals(ClassConstants.NAME_JAVA_LANG_REFLECT_METHOD))
        {
            String methodName = methodrefConstant.getName(clazz);

            if (methodName.equals(ClassConstants.METHOD_NAME_GET_ANNOTATION)           ||
                methodName.equals(ClassConstants.METHOD_NAME_GET_ANNOTATIONS)          ||
                methodName.equals(ClassConstants.METHOD_NAME_GET_DECLARED_ANNOTATIONS) ||
                methodName.equals(ClassConstants.METHOD_NAME_GET_PARAMETER_ANNOTATIONS))
            {
                notePrinter.print(clazz.getName(),
                                  "Note: " +
                                  ClassUtil.externalClassName(clazz.getName()) +
                                  " calls '" +
                                  ClassUtil.externalShortClassName(ClassUtil.externalClassName(className)) +
                                  "." +
                                  methodName + "'");
            }
        }
    }
}
