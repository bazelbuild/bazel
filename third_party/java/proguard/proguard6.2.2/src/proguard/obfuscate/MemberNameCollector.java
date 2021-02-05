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
import proguard.classfile.util.*;
import proguard.classfile.visitor.MemberVisitor;

import java.util.Map;

/**
 * This MemberVisitor collects all new (obfuscation) names of the members
 * that it visits.
 *
 * @see MemberObfuscator
 *
 * @author Eric Lafortune
 */
public class MemberNameCollector
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final boolean allowAggressiveOverloading;
    private final Map     descriptorMap;


    /**
     * Creates a new MemberNameCollector.
     * @param allowAggressiveOverloading a flag that specifies whether class
     *                                   members can be overloaded aggressively.
     * @param descriptorMap              the map of descriptors to
     *                                   [new name - old name] maps.
     */
    public MemberNameCollector(boolean allowAggressiveOverloading,
                               Map     descriptorMap)
    {
        this.allowAggressiveOverloading = allowAggressiveOverloading;
        this.descriptorMap              = descriptorMap;
    }


    // Implementations for MemberVisitor.

    public void visitAnyMember(Clazz clazz, Member member)
    {
        // Special cases: <clinit> and <init> are always kept unchanged.
        // We can ignore them here.
        String name = member.getName(clazz);
        if (ClassUtil.isInitializer(name))
        {
            return;
        }

        // Get the member's new name.
        String newName = MemberObfuscator.newMemberName(member);

        // Remember it, if it has already been set.
        if (newName != null)
        {
            // Get the member's descriptor.
            String descriptor = member.getDescriptor(clazz);

            // Check whether we're allowed to do aggressive overloading
            if (!allowAggressiveOverloading)
            {
                // Trim the return argument from the descriptor if not.
                // Works for fields and methods alike.
                descriptor = descriptor.substring(0, descriptor.indexOf(')')+1);
            }

            // Put the [descriptor - new name] in the map,
            // creating a new [new name - old name] map if necessary.
            Map nameMap = MemberObfuscator.retrieveNameMap(descriptorMap, descriptor);

            // Isn't there another original name for this new name, or should
            // this original name get priority?
            String otherName = (String)nameMap.get(newName);
            if (otherName == null                              ||
                MemberObfuscator.hasFixedNewMemberName(member) ||
                name.compareTo(otherName) < 0)
            {
                // Remember not to use the new name again in this name space.
                nameMap.put(newName, name);
            }
        }
    }
}
