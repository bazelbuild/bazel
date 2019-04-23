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
package proguard.classfile.attribute;

import proguard.classfile.Clazz;
import proguard.classfile.attribute.visitor.*;

/**
 * This Attribute represents a bootstrap methods attribute.
 *
 * @author Eric Lafortune
 */
public class BootstrapMethodsAttribute extends Attribute
{
    public int                   u2bootstrapMethodsCount;
    public BootstrapMethodInfo[] bootstrapMethods;


    /**
     * Creates an uninitialized BootstrapMethodsAttribute.
     */
    public BootstrapMethodsAttribute()
    {
    }


    /**
     * Creates an initialized BootstrapMethodsAttribute.
     */
    public BootstrapMethodsAttribute(int                   u2attributeNameIndex,
                                     int                   u2bootstrapMethodsCount,
                                     BootstrapMethodInfo[] bootstrapMethods)
    {
        super(u2attributeNameIndex);

        this.u2bootstrapMethodsCount = u2bootstrapMethodsCount;
        this.bootstrapMethods        = bootstrapMethods;
    }


    // Implementations for Attribute.

    public void accept(Clazz clazz, AttributeVisitor attributeVisitor)
    {
        attributeVisitor.visitBootstrapMethodsAttribute(clazz, this);
    }


    /**
     * Applies the given visitor to all bootstrap method info entries.
     */
    public void bootstrapMethodEntriesAccept(Clazz clazz, BootstrapMethodInfoVisitor bootstrapMethodInfoVisitor)
    {
        for (int index = 0; index < u2bootstrapMethodsCount; index++)
        {
            // We don't need double dispatching here, since there is only one
            // type of BootstrapMethodInfo.
            bootstrapMethodInfoVisitor.visitBootstrapMethodInfo(clazz, bootstrapMethods[index]);
        }
    }


    /**
     * Applies the given visitor to the specified bootstrap method info
     * entry.
     */
    public void bootstrapMethodEntryAccept(Clazz                      clazz,
                                           int                        bootstrapMethodIndex,
                                           BootstrapMethodInfoVisitor bootstrapMethodInfoVisitor)
    {
        // We don't need double dispatching here, since there is only one
        // type of BootstrapMethodInfo.
        bootstrapMethodInfoVisitor.visitBootstrapMethodInfo(clazz, bootstrapMethods[bootstrapMethodIndex]);
    }
}
