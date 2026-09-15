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
 * This StringMatcher tests whether strings start with a specified variable
 * string and then match another optional given StringMatcher.
 *
 * @see VariableStringMatcher
 * @author Eric Lafortune
 */
public class MatchedStringMatcher extends StringMatcher
{
    private final VariableStringMatcher variableStringMatcher;
    private final StringMatcher         nextMatcher;


    /**
     * Creates a new MatchedStringMatcher
     *
     * @param variableStringMatcher the variable string matcher that can
     *                              provide the string to match.
     * @param nextMatcher           an optional string matcher to match the
     *                              remainder of the string.
     */
    public MatchedStringMatcher(VariableStringMatcher variableStringMatcher,
                                StringMatcher         nextMatcher)
    {
        this.variableStringMatcher = variableStringMatcher;
        this.nextMatcher           = nextMatcher;
    }


    // Implementation for StringMatcher.

    @Override
    protected boolean matches(String string, int beginOffset, int endOffset)
    {
        String matchingString = variableStringMatcher.getMatchingString();
        if (matchingString == null)
        {
            return false;
        }

        int stringLength        = endOffset - beginOffset;
        int matchngStringLength = matchingString.length();
        return stringLength >= matchngStringLength &&
               string.startsWith(matchingString, beginOffset) &&
               ((nextMatcher == null && stringLength == matchngStringLength) ||
                nextMatcher.matches(string,
                                    beginOffset + matchngStringLength,
                                    endOffset));
    }
}
