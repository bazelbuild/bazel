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
 * This interface provides methods to determine whether strings match a given
 * criterion, which is specified by the implementation.
 *
 * @author Eric Lafortune
 */
public abstract class StringMatcher
{
    /**
     * Checks whether the given string matches.
     * @param string the string to match.
     * @return a boolean indicating whether the string matches the criterion.
     */
    public boolean matches(String string)
    {
        return matches(string, 0, string.length());
    }


    /**
     * Checks whether the given substring matches.
     * @param string the string to match.
     * @param beginOffset the start offset of the substring (inclusive).
     * @param endOffset the end offset of the substring (exclusive).
     * @return a boolean indicating whether the substring matches the criterion.
     */
    protected abstract boolean matches(String string,
                                       int    beginOffset,
                                       int    endOffset);
}
