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


import proguard.classfile.attribute.Attribute;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.visitor.MemberVisitor;

/**
 * Representation of a field or method from a program class.
 *
 * @author Eric Lafortune
 */
public abstract class ProgramMember implements Member
{
    public int         u2accessFlags;
    public int         u2nameIndex;
    public int         u2descriptorIndex;
    public int         u2attributesCount;
    public Attribute[] attributes;

    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Creates an uninitialized ProgramMember.
     */
    protected ProgramMember()
    {
    }


    /**
     * Creates an initialized ProgramMember.
     */
    protected ProgramMember(int         u2accessFlags,
                            int         u2nameIndex,
                            int         u2descriptorIndex,
                            int         u2attributesCount,
                            Attribute[] attributes)
    {
        this.u2accessFlags     = u2accessFlags;
        this.u2nameIndex       = u2nameIndex;
        this.u2descriptorIndex = u2descriptorIndex;
        this.u2attributesCount = u2attributesCount;
        this.attributes        = attributes;
    }


    /**
     * Returns the (first) attribute with the given name.
     */
    private Attribute getAttribute(Clazz clazz, String name)
    {
        for (int index = 0; index < u2attributesCount; index++)
        {
            Attribute attribute = attributes[index];
            if (attribute.getAttributeName(clazz).equals(name))
            {
                return attribute;
            }
        }

        return null;
    }


    /**
     * Accepts the given member info visitor.
     */
    public abstract void accept(ProgramClass  programClass,
                                MemberVisitor memberVisitor);



    /**
     * Lets the given attribute info visitor visit all the attributes of
     * this member info.
     */
    public abstract void attributesAccept(ProgramClass     programClass,
                                          AttributeVisitor attributeVisitor);


    // Implementations for Member.

    public int getAccessFlags()
    {
        return u2accessFlags;
    }

    public String getName(Clazz clazz)
    {
        return clazz.getString(u2nameIndex);
    }

    public String getDescriptor(Clazz clazz)
    {
        return clazz.getString(u2descriptorIndex);
    }

    public void accept(Clazz clazz, MemberVisitor memberVisitor)
    {
        accept((ProgramClass)clazz, memberVisitor);
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
