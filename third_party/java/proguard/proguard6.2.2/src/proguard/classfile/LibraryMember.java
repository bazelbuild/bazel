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

import proguard.classfile.visitor.MemberVisitor;

/**
 * Representation of a field or method from a library class.
 *
 * @author Eric Lafortune
 */
public abstract class LibraryMember implements Member
{
    public int    u2accessFlags;
    public String name;
    public String descriptor;

    /**
     * An extra field in which visitors can store information.
     */
    public Object visitorInfo;


    /**
     * Creates an uninitialized LibraryMember.
     */
    protected LibraryMember()
    {
    }


    /**
     * Creates an initialized LibraryMember.
     */
    protected LibraryMember(int    u2accessFlags,
                            String name,
                            String descriptor)
    {
        this.u2accessFlags = u2accessFlags;
        this.name          = name;
        this.descriptor    = descriptor;
    }


    /**
     * Accepts the given member info visitor.
     */
    public abstract void accept(LibraryClass  libraryClass,
                                MemberVisitor memberVisitor);


    // Implementations for Member.

    public int getAccessFlags()
    {
        return u2accessFlags;
    }

    public String getName(Clazz clazz)
    {
        return name;
    }

    public String getDescriptor(Clazz clazz)
    {
        return descriptor;
    }

    public void accept(Clazz clazz, MemberVisitor memberVisitor)
    {
        accept((LibraryClass)clazz, memberVisitor);
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
