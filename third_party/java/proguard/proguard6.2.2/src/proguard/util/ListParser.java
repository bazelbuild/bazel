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

import java.util.List;

/**
 * This StringParser can create StringMatcher instances for regular expressions.
 * The regular expressions are either presented as a list, or they are
 * interpreted as comma-separated lists, optionally prefixed with '!' negators.
 * If an entry with a negator matches, a negative match is returned, without
 * considering any subsequent entries in the list. The creation of StringMatcher
 * instances  for the entries is delegated to the given StringParser.
 *
 * @author Eric Lafortune
 */
public class ListParser implements StringParser
{
    private final StringParser stringParser;


    /**
     * Creates a new ListParser that parses individual elements in the
     * comma-separated list with the given StringParser.
     */
    public ListParser(StringParser stringParser)
    {
        this.stringParser = stringParser;
    }


    // Implementations for StringParser.

    public StringMatcher parse(String regularExpression)
    {
        // Does the regular expression contain a ',' list separator?
        return parse(ListUtil.commaSeparatedList(regularExpression));
    }


    /**
     * Creates a StringMatcher for the given regular expression, which can
     * be a list of optionally negated simple entries.
     * <p>
     * An empty list results in a StringMatcher that matches any string.
     */
    public StringMatcher parse(List regularExpressions)
    {
        StringMatcher listMatcher = null;

        // Loop over all simple regular expressions, backward, creating a
        // linked list of matchers.
        for (int index = regularExpressions.size()-1; index >= 0; index--)
        {
            String regularExpression = (String)regularExpressions.get(index);

            StringMatcher entryMatcher = parseEntry(regularExpression);

            // Prepend the entry matcher.
            listMatcher =
                listMatcher == null ?
                    (StringMatcher)entryMatcher :
                isNegated(regularExpression) ?
                    (StringMatcher)new AndMatcher(entryMatcher, listMatcher) :
                    (StringMatcher)new OrMatcher(entryMatcher, listMatcher);
        }

        return listMatcher != null ? listMatcher : new ConstantMatcher(true);
    }


    // Small utility methods.

    /**
     * Creates a StringMatcher for the given regular expression, which is a
     * an optionally negated simple expression.
     */
    private StringMatcher parseEntry(String regularExpression)
    {
        // Wrap the matcher if the regular expression starts with a '!' negator.
        return isNegated(regularExpression) ?
          new NotMatcher(stringParser.parse(regularExpression.substring(1))) :
          stringParser.parse(regularExpression);
    }


    /**
     * Returns whether the given simple regular expression is negated.
     */
    private boolean isNegated(String regularExpression)
    {
        return regularExpression.length() > 0 &&
               regularExpression.charAt(0) == '!';
    }


    /**
     * A main method for testing name matching.
     */
    public static void main(String[] args)
    {
        try
        {
            System.out.println("Regular expression ["+args[0]+"]");
            ListParser parser  = new ListParser(new NameParser());
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
