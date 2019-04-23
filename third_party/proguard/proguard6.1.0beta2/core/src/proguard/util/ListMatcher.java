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

/**
 * This StringMatcher tests whether strings match a given list of StringMatcher
 * instances. The instances are considered sequentially. Each instance in the
 * list can optionally be negated, meaning that a match makes the entire
 * remaining match fail.
 *
 * @author Eric Lafortune
 */
public class ListMatcher extends StringMatcher
{
    private final StringMatcher[] matchers;
    private final boolean[]       negate;


    public ListMatcher(StringMatcher... matchers)
    {
        this(matchers, null);
    }


    public ListMatcher(StringMatcher[] matchers, boolean[] negate)
    {
        this.matchers = matchers;
        this.negate   = negate;
    }


    // Implementations for StringMatcher.

    @Override
    protected boolean matches(String string, int beginOffset, int endOffset)
    {
        // Check the list of matchers.
        for (int index = 0; index < matchers.length; index++)
        {
            StringMatcher matcher = matchers[index];
            if (matcher.matches(string, beginOffset, endOffset))
            {
                return negate == null ||
                       !negate[index];
            }
        }

        return negate != null &&
               negate[negate.length - 1];

    }
}
