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
package proguard.obfuscate;

import proguard.classfile.*;
import proguard.classfile.attribute.Attribute;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This AttributeVisitor marks all attributes that it visits.
 *
 * @see AttributeShrinker
 *
 * @author Eric Lafortune
 */
public class AttributeUsageMarker
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    // A visitor info flag to indicate the attribute is being used.
    private static final Object USED = new Object();


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute)
    {
        markAsUsed(attribute);
    }


    // Small utility methods.

    /**
     * Marks the given VisitorAccepter as being used (or useful).
     * In this context, the VisitorAccepter will be an Attribute object.
     */
    private static void markAsUsed(VisitorAccepter visitorAccepter)
    {
        visitorAccepter.setVisitorInfo(USED);
    }


    /**
     * Returns whether the given VisitorAccepter has been marked as being used.
     * In this context, the VisitorAccepter will be an Attribute object.
     */
    static boolean isUsed(VisitorAccepter visitorAccepter)
    {
        return visitorAccepter.getVisitorInfo() == USED;
    }
}
