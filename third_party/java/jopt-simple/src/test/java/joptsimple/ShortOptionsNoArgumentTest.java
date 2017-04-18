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

import static java.util.Arrays.*;
import static java.util.Collections.*;

import org.junit.Before;
import org.junit.Test;

import static joptsimple.ParserRules.*;
import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class ShortOptionsNoArgumentTest extends AbstractOptionParserFixture {
    @Before
    public final void initializeParser() {
        parser.accepts( "a" );
        parser.accepts( "b" );
        parser.accepts( "c" );
    }

    @Test
    public void singleOption() {
        OptionSet options = parser.parse( "-a" );

        assertOptionDetected( options, "a" );
        assertNull( options.valueOf( "a" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void twoSingleOptions() {
        OptionSet options = parser.parse( "-a", "-b" );

        assertOptionDetected( options, "a" );
        assertOptionDetected( options, "b" );
        assertNull( options.valueOf( "a" ) );
        assertNull( options.valueOf( "b" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void singleOptionWithOneNonOptionArgument() {
        OptionSet options = parser.parse( "-c", "foo" );

        assertOptionDetected( options, "c" );
        assertNull( options.valueOf( "c" ) );
        assertEquals( singletonList( "foo" ), options.nonOptionArguments() );
    }

    @Test
    public void clusteredOptions() {
        OptionSet options = parser.parse( "-bac" );

        assertOptionDetected( options, "a" );
        assertOptionDetected( options, "b" );
        assertOptionDetected( options, "c" );
        assertNull( options.valueOf( "a" ) );
        assertNull( options.valueOf( "b" ) );
        assertNull( options.valueOf( "c" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void optionsTerminator() {
        OptionSet options = parser.parse( "-a", OPTION_TERMINATOR, "-a", "-b" );

        assertOptionDetected( options, "a" );
        assertNull( options.valueOf( "a" ) );
        assertOptionNotDetected( options, "b" );
        assertEquals( asList( "-a", "-b" ), options.nonOptionArguments() );
    }

    @Test
    public void appearingMultipleTimes() {
        OptionSet options = parser.parse( "-b", "-b", "-b" );

        assertOptionDetected( options, "b" );
        assertNull( options.valueOf( "b" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }
}
