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

import proguard.classfile.ClassConstants;

import java.util.*;

/**
 * This StringParser can create StringMatcher instances for regular expressions
 * matching internal class names (or descriptors containing class names).
 * The regular expressions can contain the following wildcards:
 * '%'     for a single internal primitive type character (V, Z, B, C, S, I, F,
 *         J, or D),
 * '?'     for a single regular class name character,
 * '*'     for any number of regular class name characters,
 * '**'    for any number of regular class name characters or package separator
 *         characters ('/'),
 * 'L***;' for a single internal type (class name or primitive type,
 *         array or non-array),
 * 'L///;' for any number of internal types (class names and primitive
 *         types), and
 * '<n>'   for a reference to an earlier wildcard (n = 1, 2, ...)
 *
 * @author Eric Lafortune
 */
public class ClassNameParser implements StringParser
{
    private static final char[] PRIMITIVE_TYPES = new char[]
    {
        ClassConstants.TYPE_VOID,
        ClassConstants.TYPE_BOOLEAN,
        ClassConstants.TYPE_BYTE,
        ClassConstants.TYPE_CHAR,
        ClassConstants.TYPE_SHORT,
        ClassConstants.TYPE_INT,
        ClassConstants.TYPE_LONG,
        ClassConstants.TYPE_FLOAT,
        ClassConstants.TYPE_DOUBLE,
    };


    private List variableStringMatchers;


    /**
     * Creates a new ClassNameParser.
     */
    public ClassNameParser()
    {
        this(null);
    }


    /**
     * Creates a new ClassNameParser that supports references to earlier
     * wildcards.
     *
     * @param variableStringMatchers an optional mutable list of
     *                               VariableStringMatcher instances that match
     *                               the wildcards.
     */
    public ClassNameParser(List variableStringMatchers)
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

            // Is there an 'L///;' wildcard?
            if (regularExpression.regionMatches(index, "L///;", 0, 5))
            {
                SettableMatcher settableMatcher = new SettableMatcher();

                // Create a matcher for the wildcard.
                nextMatcher = rememberVariableStringMatcher(
                    new VariableStringMatcher(null,
                                              new char[] { ClassConstants.METHOD_ARGUMENTS_CLOSE },
                                              0,
                                              Integer.MAX_VALUE,
                                              settableMatcher));

                settableMatcher.setMatcher(parse(regularExpression.substring(index + 5)));
                break;
            }

            // Is there an 'L***;' wildcard?
            else if (regularExpression.regionMatches(index, "L***;", 0, 5))
            {
                SettableMatcher settableMatcher = new SettableMatcher();

                // Create a matcher for the wildcard.
                // TODO: The returned variable matcher is actually a composite that doesn't return the entire matched string.
                nextMatcher = rememberVariableStringMatcher(
                    createAnyTypeMatcher(settableMatcher));

                // Recursively create a matcher for the rest of the string.
                settableMatcher.setMatcher(parse(regularExpression.substring(index + 5)));
                break;
            }

            // Is there a '**' wildcard?
            else if (regularExpression.regionMatches(index, "**", 0, 2))
            {
                // Handle the end of the regular expression more efficiently,
                // without any next matcher for the variable string matcher.
                SettableMatcher settableMatcher =
                    index + 2 == regularExpression.length() ? null :
                        new SettableMatcher();

                // Create a matcher for the wildcard.
                nextMatcher = rememberVariableStringMatcher(
                    new VariableStringMatcher(null,
                                              new char[] { ClassConstants.TYPE_CLASS_END },
                                              0,
                                              Integer.MAX_VALUE,
                                              settableMatcher));

                // Recursively create a matcher for the rest of the string.
                if (settableMatcher != null)
                {
                    settableMatcher.setMatcher(parse(regularExpression.substring(index + 2)));
                }
                break;
            }

            // Is there a '*' wildcard?
            else if (regularExpression.charAt(index) == '*')
            {
                SettableMatcher settableMatcher = new SettableMatcher();

                // Create a matcher for the wildcard.
                nextMatcher = rememberVariableStringMatcher(
                    new VariableStringMatcher(null,
                                              new char[] { ClassConstants.TYPE_CLASS_END, ClassConstants.PACKAGE_SEPARATOR },
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
                                              new char[] { ClassConstants.TYPE_CLASS_END, ClassConstants.PACKAGE_SEPARATOR },
                                              1,
                                              1,
                                              settableMatcher));

                // Recursively create a matcher for the rest of the string.
                settableMatcher.setMatcher(parse(regularExpression.substring(index + 1)));
                break;
            }

            // Is there a '%' wildcard?
            else if (regularExpression.charAt(index) == '%')
            {
                SettableMatcher settableMatcher = new SettableMatcher();

                // Create a matcher for the wildcard.
                nextMatcher = rememberVariableStringMatcher(
                    new VariableStringMatcher(PRIMITIVE_TYPES,
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
     * Creates a StringMatcher that matches any type (class or primitive type,
     * array or non-array) and then the given matcher.
     */
    private VariableStringMatcher createAnyTypeMatcher(StringMatcher nextMatcher)
    {
        return
            // Any number of '['.
            new VariableStringMatcher(new char[] { ClassConstants.TYPE_ARRAY },
                                      null,
                                      0,
                                      255,
            // Followed by:
            new OrMatcher(
                // A primitive type.
                new VariableStringMatcher(PRIMITIVE_TYPES,
                                          null,
                                          1,
                                          1,
                                          nextMatcher),

                // Or a class type.
                new VariableStringMatcher(new char[] { ClassConstants.TYPE_CLASS_START },
                                          null,
                                          1,
                                          1,
                new VariableStringMatcher(null,
                                          new char[] { ClassConstants.TYPE_CLASS_END },
                                          0,
                                          Integer.MAX_VALUE,
                new VariableStringMatcher(new char[] { ClassConstants.TYPE_CLASS_END },
                                          null,
                                          1,
                                          1,
                                          nextMatcher)))));
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
        return (VariableStringMatcher) variableStringMatchers.get(index);
    }


    /**
     * Parses a reference to a wildcard at the given index, if any.
     * Returns the 1-based index, or 0 otherwise.
     */
    private int wildCardIndex(String string, int index)
    throws IllegalArgumentException
    {
        if (string.charAt(index) != '<')
        {
            return 0;
        }

        int closingBracketIndex = string.indexOf('>', index);
        if (closingBracketIndex < 0)
        {
            throw new IllegalArgumentException("Missing closing angular bracket after opening bracket at index "+index+" in ["+string+"]");
        }

        if (variableStringMatchers == null)
        {
            throw new IllegalArgumentException("References to wildcards are not supported in this argument ["+string+"]");
        }

        String argumentBetweenBrackets = string.substring(index+1, closingBracketIndex);

        try
        {
            int wildcardIndex = Integer.parseInt(argumentBetweenBrackets);
            if (wildcardIndex < 1 ||
                wildcardIndex > variableStringMatchers.size())
            {
                throw new IllegalArgumentException("Invalid reference to wildcard ("+wildcardIndex+", must lie between 1 and "+variableStringMatchers.size()+" in ["+string+"])");
            }

            return wildcardIndex;
        }
        catch (NumberFormatException e)
        {
            throw new IllegalArgumentException("Reference to wildcard must be a number ["+argumentBetweenBrackets+"] in ["+string+"]");
        }
    }


    /**
     * A main method for testing class name matching.
     */
    public static void main(String[] args)
    {
        try
        {
            System.out.println("Regular expression ["+args[0]+"]");
            ClassNameParser parser  = new ClassNameParser();
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
