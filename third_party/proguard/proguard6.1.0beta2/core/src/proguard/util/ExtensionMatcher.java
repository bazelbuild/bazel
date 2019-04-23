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
 * This StringMatcher tests whether strings end in a given extension, ignoring
 * its case.
 *
 * @author Eric Lafortune
 */
public class ExtensionMatcher extends StringMatcher
{
    private final String extension;


    /**
     * Creates a new StringMatcher.
     *
     * @param extension the extension against which strings will be matched.
     */
    public ExtensionMatcher(String extension)
    {
        this.extension = extension;
    }


    // Implementations for StringMatcher.

    @Override
    protected boolean matches(String string, int beginOffset, int endOffset)
    {
        return endsWithIgnoreCase(string, beginOffset, endOffset, extension);
    }


    /**
     * Returns whether the given string ends with the given suffix, ignoring its
     * case.
     */
    private static boolean endsWithIgnoreCase(String string,
                                              int    beginOffset,
                                              int    endOffset,
                                              String suffix)
    {
        int suffixLength = suffix.length();

        return string.regionMatches(true, endOffset - suffixLength, suffix, 0, suffixLength);
    }
}
