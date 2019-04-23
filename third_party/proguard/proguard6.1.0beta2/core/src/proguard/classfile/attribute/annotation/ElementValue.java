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
package proguard.classfile.attribute.annotation;

import proguard.classfile.*;
import proguard.classfile.attribute.annotation.visitor.ElementValueVisitor;
import proguard.classfile.visitor.MemberVisitor;

/**
 * This abstract class represents an element value that is attached to an
 * annotation or an annotation default. Specific types of element values are
 * subclassed from it.
 *
 * @author Eric Lafortune
 */
public abstract class ElementValue implements VisitorAccepter
{
    /**
     * An extra field for the optional element name. It is used in element value
     * pairs of annotations. Otherwise, it is 0.
     */
    public int u2elementNameIndex;

    /**
     * An extra field pointing to the referenced <code>Clazz</code>
     * object, if applicable. This field is typically filled out by the
     * <code>{@link proguard.classfile.util.ClassReferenceInitializer}</code>.
     */
    public Clazz referencedClass;

    /**
     * An extra field pointing to the referenced <code>Method</code>
     * object, if applicable. This field is typically filled out by the
     * <code>{@link proguard.classfile.util.ClassReferenceInitializer}</code>.
     */
    public Method referencedMethod;

    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Creates an uninitialized ElementValue.
     */
    protected ElementValue()
    {
    }


    /**
     * Creates an initialized ElementValue.
     */
    protected ElementValue(int u2elementNameIndex)
    {
        this.u2elementNameIndex = u2elementNameIndex;
    }


    /**
     * Returns the element name.
     */
    public String getMethodName(Clazz clazz)
    {
        return clazz.getString(u2elementNameIndex);
    }


    // Abstract methods to be implemented by extensions.

    /**
     * Returns the tag of this element value.
     */
    public abstract char getTag();


    /**
     * Accepts the given visitor.
     */
    public abstract void accept(Clazz clazz, Annotation annotation, ElementValueVisitor elementValueVisitor);



    /**
     * Applies the given visitor to the referenced method.
     */
    public void referencedMethodAccept(MemberVisitor memberVisitor)
    {
        if (referencedMethod != null)
        {
            referencedMethod.accept(referencedClass, memberVisitor);
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
