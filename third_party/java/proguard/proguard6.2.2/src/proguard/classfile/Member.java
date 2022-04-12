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
package proguard.classfile;

import proguard.classfile.visitor.*;

/**
 * Representation of a field or method from a class.
 *
 * @author Eric Lafortune
 */
public interface Member extends VisitorAccepter
{
    /**
     * Returns the access flags.
     */
    public int getAccessFlags();

    /**
     * Returns the class member name.
     */
    public String getName(Clazz clazz);

    /**
     * Returns the class member's descriptor.
     */
    public String getDescriptor(Clazz clazz);

    /**
     * Accepts the given class visitor.
     */
    public void accept(Clazz clazz, MemberVisitor memberVisitor);

    /**
     * Lets the Clazz objects referenced in the descriptor string
     * accept the given visitor.
     */
    public void referencedClassesAccept(ClassVisitor classVisitor);
}
