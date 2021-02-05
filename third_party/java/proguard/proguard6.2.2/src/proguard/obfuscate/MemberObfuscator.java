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

import java.util.*;

/**
 * This MemberVisitor obfuscates all class members that it visits.
 * It uses names from the given name factory. At the same time, it avoids names
 * from the given descriptor map.
 * <p>
 * The class members must have been linked before applying this visitor.
 *
 * @see MethodLinker
 *
 * @author Eric Lafortune
 */
public class MemberObfuscator
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final boolean        allowAggressiveOverloading;
    private final NameFactory    nameFactory;
    private final Map            descriptorMap;


    /**
     * Creates a new MemberObfuscator.
     * @param allowAggressiveOverloading a flag that specifies whether class
     *                                   members can be overloaded aggressively.
     * @param nameFactory                the factory that can produce
     *                                   obfuscated member names.
     * @param descriptorMap              the map of descriptors to
     *                                   [new name - old name] maps.
     */
    public MemberObfuscator(boolean        allowAggressiveOverloading,
                            NameFactory    nameFactory,
                            Map            descriptorMap)
    {
        this.allowAggressiveOverloading = allowAggressiveOverloading;
        this.nameFactory                = nameFactory;
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

        // Get the member's descriptor.
        String descriptor = member.getDescriptor(clazz);

        // Check whether we're allowed to overload aggressively.
        if (!allowAggressiveOverloading)
        {
            // Trim the return argument from the descriptor if not.
            // Works for fields and methods alike.
            descriptor = descriptor.substring(0, descriptor.indexOf(')')+1);
        }

        // Get the name map, creating a new one if necessary.
        Map nameMap = retrieveNameMap(descriptorMap, descriptor);

        // Get the member's new name.
        String newName = newMemberName(member);

        // Assign a new one, if necessary.
        if (newName == null)
        {
            // Find an acceptable new name.
            nameFactory.reset();

            do
            {
                newName = nameFactory.nextName();
            }
            while (nameMap.containsKey(newName));

            // Remember not to use the new name again in this name space.
            nameMap.put(newName, name);

            // Assign the new name.
            setNewMemberName(member, newName);
        }
    }


    // Small utility methods.

    /**
     * Gets the name map, based on the given map and a given descriptor.
     * A new empty map is created if necessary.
     * @param descriptorMap the map of descriptors to [new name - old name] maps.
     * @param descriptor    the class member descriptor.
     * @return the corresponding name map.
     */
    static Map retrieveNameMap(Map descriptorMap, String descriptor)
    {
        // See if we can find the nested map with this descriptor key.
        Map nameMap = (Map)descriptorMap.get(descriptor);

        // Create a new one if not.
        if (nameMap == null)
        {
            nameMap = new HashMap();
            descriptorMap.put(descriptor, nameMap);
        }

        return nameMap;
    }


    /**
     * Assigns a fixed new name to the given class member.
     * @param member the class member.
     * @param name   the new name.
     */
    static void setFixedNewMemberName(Member member, String name)
    {
        VisitorAccepter lastVisitorAccepter = MethodLinker.lastVisitorAccepter(member);

        if (!(lastVisitorAccepter instanceof LibraryMember) &&
            !(lastVisitorAccepter instanceof MyFixedName))
        {
            lastVisitorAccepter.setVisitorInfo(new MyFixedName(name));
        }
        else
        {
            lastVisitorAccepter.setVisitorInfo(name);
        }
    }


    /**
     * Assigns a new name to the given class member.
     * @param member the class member.
     * @param name   the new name.
     */
    static void setNewMemberName(Member member, String name)
    {
        MethodLinker.lastVisitorAccepter(member).setVisitorInfo(name);
    }


    /**
     * Returns whether the new name of the given class member is fixed.
     * @param member the class member.
     * @return whether its new name is fixed.
     */
    static boolean hasFixedNewMemberName(Member member)
    {
        VisitorAccepter lastVisitorAccepter = MethodLinker.lastVisitorAccepter(member);

        return lastVisitorAccepter instanceof LibraryMember ||
               lastVisitorAccepter instanceof MyFixedName;
    }


    /**
     * Retrieves the new name of the given class member.
     * @param member the class member.
     * @return the class member's new name, or <code>null</code> if it doesn't
     *         have one yet.
     */
    static String newMemberName(Member member)
    {
        return (String)MethodLinker.lastVisitorAccepter(member).getVisitorInfo();
    }


    /**
     * This VisitorAccepter can be used to wrap a name string, to indicate that
     * the name is fixed.
     */
    private static class MyFixedName implements VisitorAccepter
    {
        private String newName;


        public MyFixedName(String newName)
        {
            this.newName = newName;
        }


        // Implementations for VisitorAccepter.

        public Object getVisitorInfo()
        {
            return newName;
        }


        public void setVisitorInfo(Object visitorInfo)
        {
            newName = (String)visitorInfo;
        }
    }
}
