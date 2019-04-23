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
package proguard.classfile.visitor;

import proguard.classfile.*;
import proguard.classfile.util.SimplifiedVisitor;

import java.util.Set;

/**
 * This MemberVisitor collects dot-separated classname.membername.descriptor
 * strings of the class members that it visits.
 *
 * @author Eric Lafortune
 */
public class MemberCollector
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final boolean includeClassName;
    private final boolean includeMemberName;
    private final boolean includeMemberDescriptor;

    private final Set set;


    /**
     * Creates a new MemberCollector.
     * @param includeClassName        specifies whether to include the class
     *                                name in each collected strings.
     * @param includeMemberName       specifies whether to include the member
     *                                name in each collected strings.
     * @param includeMemberDescriptor specifies whether to include the member
     *                                descriptor in each collected strings.
     * @param set                     the Set in which all strings will be
     *                                collected.
     */
    public MemberCollector(boolean includeClassName,
                           boolean includeMemberName,
                           boolean includeMemberDescriptor,
                           Set     set)
    {
        this.includeClassName        = includeClassName;
        this.includeMemberName       = includeMemberName;
        this.includeMemberDescriptor = includeMemberDescriptor;

        this.set = set;
    }


    // Implementations for MemberVisitor.

    public void visitAnyMember(Clazz clazz, Member member)
    {
        StringBuffer buffer = new StringBuffer();

        if (includeClassName)
        {
            buffer.append(clazz.getName()).append('.');
        }

        if (includeMemberName)
        {
            buffer.append(member.getName(clazz)).append('.');
        }

        if (includeMemberDescriptor)
        {
            buffer.append(member.getDescriptor(clazz));
        }

        set.add(buffer.toString());
    }
}