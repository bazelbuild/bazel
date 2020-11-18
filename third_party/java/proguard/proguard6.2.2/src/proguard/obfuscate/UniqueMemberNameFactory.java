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

import proguard.classfile.Clazz;

/**
 * NameFactory which only generates names that don't exist yet as members
 * on the class for which it is created.
 *
 * @author Johan Leys
 */
public class UniqueMemberNameFactory implements NameFactory
{
    private static final String INJECTED_MEMBER_PREFIX = "$$";

    private final NameFactory delegateNameFactory;
    private final Clazz       clazz;


    /**
     * Utility for creating a new NameFactory that can generate names for injected
     * members: the generated names are unique within the given class, and don't
     * clash with non-injected members of its super classes.
     *
     * @param clazz the class for which to generate a NameFactory.
     * @return the new NameFactory instance.
     */
    public static UniqueMemberNameFactory newInjectedMemberNameFactory(Clazz clazz)
    {
        return new UniqueMemberNameFactory(
               new PrefixingNameFactory(
               new SimpleNameFactory(), INJECTED_MEMBER_PREFIX), clazz);
    }


    /**
     * Creates a new UniqueMemberNameFactory.
     * @param delegateNameFactory the delegate NameFactory, used for generating
     *                            new candidate names.
     * @param clazz               the class in which to check for existing
     *                            member names.
     */
    public UniqueMemberNameFactory(NameFactory delegateNameFactory,
                                   Clazz       clazz)
    {
        this.delegateNameFactory = delegateNameFactory;
        this.clazz               = clazz;
    }


    // Implementations for NameFactory.

    public String nextName()
    {
        String name;

        // Check if the name doesn't exist yet. We don't have additional
        // descriptor information, so we can only search on the name.
        do
        {
            name = delegateNameFactory.nextName();
        }
        while (clazz.findField(name, null) != null ||
               clazz.findMethod(name, null) != null);

        return name;
    }


    public void reset()
    {
        delegateNameFactory.reset();
    }
}