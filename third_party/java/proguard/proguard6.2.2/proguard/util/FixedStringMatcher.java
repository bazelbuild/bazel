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
 * This StringMatcher tests whether strings start with a given fixed string
 * and then match another optional given StringMatcher.
 *
 * @author Eric Lafortune
 */
public class FixedStringMatcher extends StringMatcher
{
    private final String        fixedString;
    private final StringMatcher nextMatcher;


    /**
     * Creates a new FixedStringMatcher.
     *
     * @param fixedString the string to match.
     */
    public FixedStringMatcher(String fixedString)
    {
        this(fixedString, null);
    }


    /**
     * Creates a new FixedStringMatcher.
     *
     * @param fixedString the string prefix to match.
     * @param nextMatcher an optional string matcher to match the remainder of
     *                    the string.
     */
    public FixedStringMatcher(String fixedString, StringMatcher nextMatcher)
    {
        this.fixedString = fixedString;
        this.nextMatcher = nextMatcher;
    }


    // Implementations for StringMatcher.

    @Override
    protected boolean matches(String string, int beginOffset, int endOffset)
    {
        int stringLength      = endOffset - beginOffset;
        int fixedStringLength = fixedString.length();
        return stringLength >= fixedStringLength &&
               string.startsWith(fixedString, beginOffset) &&
               ((nextMatcher == null && stringLength == fixedStringLength) ||
                nextMatcher.matches(string,
                                    beginOffset + fixedStringLength,
                                    endOffset));
    }
}
