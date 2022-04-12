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

import proguard.classfile.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;

/**
 * This abstract class represents an attribute that is attached to a class,
 * a class member, or a code attribute. Specific types of attributes are
 * subclassed from it.
 *
 * @author Eric Lafortune
 * @noinspection AbstractClassWithoutAbstractMethods
 */
public abstract class Attribute implements VisitorAccepter
{
    public int u2attributeNameIndex;
    //public int  u4attributeLength;
    //public byte info[];

    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Create an uninitialized Attribute.
     */
    protected Attribute()
    {
    }


    /**
     * Create an initialized Attribute.
     */
    protected Attribute(int u2attributeNameIndex)
    {
        this.u2attributeNameIndex = u2attributeNameIndex;
    }


    /**
     * Returns the String name of the attribute.
     */
    public String getAttributeName(Clazz clazz)
    {
        return clazz.getString(u2attributeNameIndex);
    }


    // Methods to be implemented by extensions, if applicable.

    /**
     * Accepts the given visitor.
     */
    public void accept(Clazz clazz, AttributeVisitor attributeVisitor)
    {
        throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
    }

    /**
     * Accepts the given visitor in the context of the given field.
     */
    public void accept(Clazz clazz, Field field, AttributeVisitor attributeVisitor)
    {
        // Delegate to the default invocation if the field is null anyway.
        if (field == null)
        {
            accept(clazz, attributeVisitor);
        }
        else
        {
            throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
        }
    }

    /**
     * Accepts the given visitor in the context of the given method.
     */
    public void accept(Clazz clazz, Method method, AttributeVisitor attributeVisitor)
    {
        // Delegate to the default invocation if the method is null anyway.
        if (method == null)
        {
            accept(clazz, (Field)null, attributeVisitor);
        }
        else
        {
            throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
        }
    }

    /**
     * Accepts the given visitor in the context of the given code attribute.
     */
    public void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, AttributeVisitor attributeVisitor)
    {
        // Delegate to the default invocation if the code attribute is null
        // anyway.
        if (codeAttribute == null)
        {
            accept(clazz, method, attributeVisitor);
        }
        else
        {
            throw new UnsupportedOperationException("Method must be overridden in ["+this.getClass().getName()+"] if ever called");
        }
    }


    // Implementations for VisitorAccepter.

    public Object getVisitorInfo()
    {
        return visitorInfo;
    }

    public void setVisitorInfo(Object visitorInfo)
    {
        this.visitorInfo = visitorInfo;
    }
}
