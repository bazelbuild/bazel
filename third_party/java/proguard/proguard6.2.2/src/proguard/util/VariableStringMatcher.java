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
 * @author Eric Lafortune
 */
public class VariableStringMatcher extends StringMatcher
{
    private final char[]        allowedCharacters;
    private final char[]        disallowedCharacters;
    private final int           minimumLength;
    private final int           maximumLength;
    private final StringMatcher nextMatcher;

    // Remember the most recently attempted match.
    private String string;
    private int    matchingBeginOffset;
    private int    matchingEndOffset;
    private String matchingString;


    /**
     * Creates a new VariableStringMatcher.
     *
     * @param allowedCharacters    an optional list of allowed characters.
     * @param disallowedCharacters an optional list of disallowed characters.
     * @param minimumLength        the minimum length of te variable string.
     * @param maximumLength        the maximum length of te variable string.
     * @param nextMatcher          an optional next matcher for the remainder
     *                             of the string.
     */
    public VariableStringMatcher(char[]        allowedCharacters,
                                 char[]        disallowedCharacters,
                                 int           minimumLength,
                                 int           maximumLength,
                                 StringMatcher nextMatcher)
    {
        this.allowedCharacters    = allowedCharacters;
        this.disallowedCharacters = disallowedCharacters;
        this.minimumLength        = minimumLength;
        this.maximumLength        = maximumLength;
        this.nextMatcher          = nextMatcher;
    }


    /**
     * Returns the string that has been matched most recently.
     */
    public String getMatchingString()
    {
        if (string == null)
        {
            throw new UnsupportedOperationException("Can't handle regular expression with referenced wildcard");
        }

        // Cache the matching string, since String#substring has become
        // expensive since about JRE 1.7.
        return matchingString == null ?
            (matchingString = string.substring(matchingBeginOffset, matchingEndOffset)) :
            matchingString;
    }


    // Implementations for StringMatcher.

    @Override
    protected boolean matches(String string, int beginOffset, int endOffset)
    {
        int length = endOffset - beginOffset;

        // Handle the case without next matcher more efficiently.
        if (nextMatcher == null)
        {
            // Check the length and the characters.
            boolean match =
                length >= minimumLength &&
                length <= maximumLength &&
                areAllowedCharacters(string, beginOffset, endOffset);

            // Does the string match?
            if (match)
            {
                // Remember the matching string, so subsequent external
                // matchers can use it.
                rememberMatchingString(string, beginOffset, endOffset);
            }
            else
            {
                // Reset the matching string, so subsequent external
                // matchers cannot use it.
                resetMatchingString();
            }

            return match;
        }

        // Check the minimum length and the corresponding characters.
        if (length < minimumLength ||
            !areAllowedCharacters(string, beginOffset, beginOffset + minimumLength))
        {
            // Reset the matching string, so subsequent external
            // matchers cannot use it.
            resetMatchingString();

            return false;
        }

        int maximumLength = Math.min(this.maximumLength, length);

        // Check the remaining characters, up to the maximum number.
        for (int index = minimumLength; index < maximumLength; index++)
        {
            int offset = beginOffset + index;

            // Check the next matcher.
            if (matchesNextMatcher(string, beginOffset, offset, endOffset))
            {
                return true;
            }

            // Otherwise just check the next character.
            if (!isAllowedCharacter(string.charAt(offset)))
            {
                // Reset the matching string, so subsequent external
                // matchers cannot use it.
                resetMatchingString();

                return false;
            }
        }

        // Last try: check the remaining characters in the string.
        int offset = beginOffset + maximumLength;

        // Check the next matcher.
        if (matchesNextMatcher(string, beginOffset, offset, endOffset))
        {
            return true;
        }

        // Reset the matching string, so subsequent external
        // matchers cannot use it.
        resetMatchingString();

        return false;
    }


    // Small utility methods.

    /**
     * Returns whether the character characters in the specified substring are
     * allowed.
     */
    private boolean areAllowedCharacters(String string, int beginOffset, int endOffset)
    {
        for (int offset = beginOffset; offset < endOffset; offset++)
        {
            if (!isAllowedCharacter(string.charAt(offset)))
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Returns whether the given character is allowed in the variable string.
     */
    private boolean isAllowedCharacter(char character)
    {
        // Check the allowed characters.
        if (allowedCharacters != null)
        {
            for (int index = 0; index < allowedCharacters.length; index++)
            {
                if (allowedCharacters[index] == character)
                {
                    return true;
                }
            }

            return false;
        }

        // Check the disallowed characters.
        if (disallowedCharacters != null)
        {
            for (int index = 0; index < disallowedCharacters.length; index++)
            {
                if (disallowedCharacters[index] == character)
                {
                    return false;
                }
            }
        }

        // Any remaining character is allowed.
        return true;
    }


    /**
     * Returns whether the next matcher matches the specified second part of
     * the string, remembering the first part that matchers this matcher.
     */
    private boolean matchesNextMatcher(String string,
                                       int    beginOffset,
                                       int    splitOffset,
                                       int    endOffset)
    {
        // Remember the matching string, so the next matchers can use it.
        rememberMatchingString(string, beginOffset, splitOffset);

        // Check the next matcher.
        return nextMatcher.matches(string, splitOffset, endOffset);
    }


    /**
     * Remembers the string that is currently being matched.
     */
    private void rememberMatchingString(String string,
                                        int    matchingBeginOffset,
                                        int    matchingEndOffset)
    {
        this.string              = string;
        this.matchingBeginOffset = matchingBeginOffset;
        this.matchingEndOffset   = matchingEndOffset;
        this.matchingString      = null;
    }


    /**
     * Resets the string that is currently being matched.
     */
    private void resetMatchingString()
    {
        this.string              = null;
        this.matchingBeginOffset = 0;
        this.matchingEndOffset   = 0;
        this.matchingString      = null;
    }
}
