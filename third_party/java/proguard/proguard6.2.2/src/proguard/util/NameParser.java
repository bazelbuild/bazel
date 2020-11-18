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

import java.util.*;

/**
 * This StringParser can create StringMatcher instances for regular expressions
 * matching names. The regular expressions are interpreted as comma-separated
 * lists of names, optionally prefixed with '!' negators.
 * If a name with a negator matches, a negative match is returned, without
 * considering any subsequent entries in the list.
 * The regular expressions can contain the following wildcards:
 * '?'   for a single character,
 * '*'   for any number of characters, and
 * '<n>' for a reference to an earlier wildcard (n = 1, 2, ...)
 *
 * @author Eric Lafortune
 */
public class NameParser implements StringParser
{
    private List variableStringMatchers;


    /**
     * Creates a new NameParser.
     */
    public NameParser()
    {
        this(null);
    }


    /**
     * Creates a new NameParser that supports references to earlier
     * wildcards.
     *
     * @param variableStringMatchers an optional mutable list of
     *                               VariableStringMatcher instances that match
     *                               the wildcards.
     */
    public NameParser(List variableStringMatchers)
    {
        this.variableStringMatchers = variableStringMatchers;
    }


    // Implementations for StringParser.

    public StringMatcher parse(String regularExpression)
    {
        int           index;
        StringMatcher nextMatcher = new EmptyStringMatcher();

        // Look for wildcards.
        for (index = 0; index < regularExpression.length(); index++)
        {
            int wildCardIndex;

            // Is there a '*' wildcard?
            if (regularExpression.charAt(index) == '*')
            {
                SettableMatcher settableMatcher = new SettableMatcher();

                // Create a matcher for the wildcard.
                nextMatcher = rememberVariableStringMatcher(
                    new VariableStringMatcher(null,
                                              null,
                                              0,
                                              Integer.MAX_VALUE,
                                              settableMatcher));

                // Recursively create a matcher for the rest of the string.
                settableMatcher.setMatcher(parse(regularExpression.substring(index + 1)));
                break;
            }

            // Is there a '?' wildcard?
            else if (regularExpression.charAt(index) == '?')
            {
                SettableMatcher settableMatcher = new SettableMatcher();

                // Create a matcher for the wildcard.
                nextMatcher = rememberVariableStringMatcher(
                    new VariableStringMatcher(null,
                                              null,
                                              1,
                                              1,
                                              settableMatcher));

                // Recursively create a matcher for the rest of the string.
                settableMatcher.setMatcher(parse(regularExpression.substring(index + 1)));
                break;
            }

            // Is there a '<n>' wildcard?
            else if ((wildCardIndex = wildCardIndex(regularExpression, index)) > 0)
            {
                // Find the index of the closing bracket again.
                int closingIndex = regularExpression.indexOf('>', index + 1);

                // Retrieve the specified variable string matcher and
                // recursively create a matcher for the rest of the string.
                nextMatcher =
                    new MatchedStringMatcher(retrieveVariableStringMatcher(wildCardIndex - 1),
                                             parse(regularExpression.substring(closingIndex + 1)));
                break;
            }
        }

        // Return a matcher for the fixed first part of the regular expression,
        // if any, and the remainder.
        return index != 0 ?
            (StringMatcher)new FixedStringMatcher(regularExpression.substring(0, index), nextMatcher) :
            (StringMatcher)nextMatcher;
    }


    // Small utility methods.

    /**
     * Parses a reference to a wildcard at the given index, if any.
     * Returns the 1-based index, or 0 otherwise.
     */
    private int wildCardIndex(String string, int index)
    throws IllegalArgumentException
    {
        if (variableStringMatchers == null ||
            string.charAt(index) != '<')
        {
            return 0;
        }

        int closingBracketIndex = string.indexOf('>', index);
        if (closingBracketIndex < 0)
        {
            throw new IllegalArgumentException("Missing closing angular bracket");
        }

        String argumentBetweenBrackets = string.substring(index+1, closingBracketIndex);

        try
        {
            int wildcardIndex = Integer.parseInt(argumentBetweenBrackets);
            if (wildcardIndex < 1 ||
                wildcardIndex > variableStringMatchers.size())
            {
                throw new IllegalArgumentException("Invalid reference to wildcard ("+wildcardIndex+", must lie between 1 and "+variableStringMatchers.size()+")");
            }

            return wildcardIndex;
        }
        catch (NumberFormatException e)
        {
            return 0;
        }
    }


    /**
     * Adds the given variable string matcher to the list of string matchers.
     */
    private VariableStringMatcher rememberVariableStringMatcher(VariableStringMatcher variableStringMatcher)
    {
        if (variableStringMatchers != null)
        {
            variableStringMatchers.add(variableStringMatcher);
        }

        return variableStringMatcher;
    }


    /**
     * Retrieves the specified variable string matcher from the list of string
     * matchers.
     */
    private VariableStringMatcher retrieveVariableStringMatcher(int index)
    {
        return (VariableStringMatcher)variableStringMatchers.get(index);
    }


    /**
     * A main method for testing name matching.
     */
    public static void main(String[] args)
    {
        try
        {
            System.out.println("Regular expression ["+args[0]+"]");
            NameParser parser  = new NameParser();
            StringMatcher  matcher = parser.parse(args[0]);
            for (int index = 1; index < args.length; index++)
            {
                String string = args[index];
                System.out.print("String             ["+string+"]");
                System.out.println(" -> match = "+matcher.matches(args[index]));
            }
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
}
