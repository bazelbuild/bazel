/*
 The MIT License

 Copyright (c) 2004-2015 Paul R. Holser, Jr.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

package joptsimple;

import java.util.List;

import static java.util.Arrays.*;
import static java.util.Collections.*;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class OptionSynonymRequiredArgumentTest extends AbstractOptionParserFixture {
    private String optionArgument;

    @Before
    public final void initializeParser() {
        parser.acceptsAll( asList( "N", "after-date", "newer" ), "date" ).withRequiredArg().ofType( Integer.class );
        optionArgument = "2000";
    }

    @Test
    public void hasAllSynonymsWhenFirstSynonymParsed() {
        assertDetections( new String[] { "-N", optionArgument }, singletonList( Integer.valueOf( optionArgument ) ) );
    }

    @Test
    public void hasAllSynonymsWhenSecondSynonymParsed() {
        assertDetections(
            new String[] { "--after-date", optionArgument },
            singletonList( Integer.valueOf( optionArgument ) ) );
    }

    @Test
    public void hasAllSynonymsWhenThirdSynonymParsed() {
        assertDetections(
            new String[] { "--newer", optionArgument },
            singletonList( Integer.valueOf( optionArgument ) ) );
    }

    @Test
    public void reportsSameListOfArgumentsForEverySynonymOption() {
        assertDetections( new String[] { "-N", "1", "-aft", "2", "--ne", "3" }, asList( 1, 2, 3 ) );
    }

    private void assertDetections( String[] args, List<?> optionArguments ) {
        OptionSet options = parser.parse( args );

        assertEquals( optionArguments, options.valuesOf( "N" ) );
        assertEquals( optionArguments, options.valuesOf( "after-date" ) );
        assertEquals( optionArguments, options.valuesOf( "newer" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }
}
