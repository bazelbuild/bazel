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
package proguard.util;

import java.util.Set;

/**
 * This matcher tests whether strings match with a String in a given Set.
 *
 * @author Johan Leys
 */
public class CollectionMatcher extends StringMatcher
{
    private final Set<String> set;


    public CollectionMatcher(Set<String> set)
    {
        this.set = set;
    }


    // Implementations for StringMatcher.

    @Override
    public boolean matches(String string)
    {
        return set.contains(string);
    }


    @Override
    protected boolean matches(String string, int beginOffset, int endOffset)
    {
        return set.contains(string.substring(beginOffset, endOffset));
    }
}
