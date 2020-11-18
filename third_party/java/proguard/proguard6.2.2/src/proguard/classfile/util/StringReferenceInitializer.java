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
package proguard.classfile.util;

import proguard.classfile.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;

/**
 * This ConstantVisitor initializes any class references of all string constants
 * it visits. More specifically, it fills out the references of string constant
 * pool entries that happen to refer to a class in the program class pool or in
 * the library class pool.
 *
 * @author Eric Lafortune
 */
public class StringReferenceInitializer
extends      SimplifiedVisitor
implements   ConstantVisitor
{
    private final ClassPool programClassPool;
    private final ClassPool libraryClassPool;


    /**
     * Creates a new StringReferenceInitializer.
     */
    public StringReferenceInitializer(ClassPool programClassPool,
                                      ClassPool libraryClassPool)
    {
        this.programClassPool = programClassPool;
        this.libraryClassPool = libraryClassPool;
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        if (stringConstant.referencedClass == null)
        {
            // See if we can find the referenced class.
            stringConstant.referencedClass =
                findClass(ClassUtil.internalClassName(
                          ClassUtil.externalBaseType(stringConstant.getString(clazz))));
        }
    }


    // Small utility methods.

    /**
     * Returns the class with the given name, either for the program class pool
     * or from the library class pool, or <code>null</code> if it can't be found.
     */
    private Clazz findClass(String name)
    {
        // First look for the class in the program class pool.
        Clazz clazz = programClassPool.getClass(name);

        // Otherwise look for the class in the library class pool.
        if (clazz == null)
        {
            clazz = libraryClassPool.getClass(name);
        }

        return clazz;
    }
}