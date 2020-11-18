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
import proguard.classfile.constant.ClassConstant;
import proguard.evaluation.value.ValueFactory;

/**
 * This class creates java.lang.Class ReferenceValue instances that correspond
 * to specified constant pool entries.
 *
 * @author Eric Lafortune
 */
public class ClassConstantValueFactory
extends      ConstantValueFactory
{
    public ClassConstantValueFactory(ValueFactory valueFactory)
    {
        super(valueFactory);
    }


    // Implementations for ConstantVisitor.

    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Create a Class reference instead of a reference to the class.
        value = valueFactory.createReferenceValue(ClassConstants.NAME_JAVA_LANG_CLASS,
                                                  classConstant.javaLangClassClass,
                                                  false,
                                                  false);
    }
}
