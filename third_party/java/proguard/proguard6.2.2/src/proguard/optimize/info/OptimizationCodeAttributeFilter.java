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
package proguard.optimize.info;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.optimize.KeepMarker;

/**
 * This AttributeVisitor delegates calls for code attributes to another
 * AttributeVisitor, but only if they can be optimized.
 * <p>
 * <b>Note:</b> any other attribute will <b>not</b> be delegated.
 * </p>
 *
 * @author Thomas Neidhart
 */
public class OptimizationCodeAttributeFilter
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    private final AttributeVisitor attributeVisitor;
    private final AttributeVisitor otherAttributeVisitor;


    /**
     * Creates a new OptimizationCodeAttributeFilter.
     * @param attributeVisitor the <code>AttributeVisitor</code> to which visits will
     *                         be delegated.
     */
    public OptimizationCodeAttributeFilter(AttributeVisitor attributeVisitor)
    {
        this(attributeVisitor, null);
    }


    /**
     * Creates a new OptimizationCodeAttributeFilter.
     * @param attributeVisitor      the <code>AttributeVisitor</code> to which visits will
     *                              be delegated if the code attribute can be optimized.
     * @param otherAttributeVisitor the <code>AttributeVisitor</code> to which visits will
     *                              be delegated if the code attribute must be kept.
     */
    public OptimizationCodeAttributeFilter(AttributeVisitor attributeVisitor,
                                           AttributeVisitor otherAttributeVisitor)
    {
        this.attributeVisitor      = attributeVisitor;
        this.otherAttributeVisitor = otherAttributeVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        AttributeVisitor visitor = !KeepMarker.isKept(codeAttribute) ?
            attributeVisitor : otherAttributeVisitor;

        if (visitor != null)
        {
            visitor.visitCodeAttribute(clazz, method, codeAttribute);
        }
    }

}