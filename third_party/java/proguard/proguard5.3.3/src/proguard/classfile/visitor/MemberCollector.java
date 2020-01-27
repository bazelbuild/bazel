/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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
 * This MemberVisitor collects the concatenated name/descriptor strings of
 * class members that have been visited.
 *
 * @author Eric Lafortune
 */
public class MemberCollector
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final Set set;


    /**
     * Creates a new MemberCollector.
     * @param set the <code>Set</code> in which all method names/descriptor
     *            strings will be collected.
     */
    public MemberCollector(Set set)
    {
        this.set = set;
    }


    // Implementations for MemberVisitor.


    public void visitAnyMember(Clazz clazz, Member member)
    {
        set.add(member.getName(clazz) + member.getDescriptor(clazz));
    }
}