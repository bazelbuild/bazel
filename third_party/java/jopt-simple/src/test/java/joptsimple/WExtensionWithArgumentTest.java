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

import static java.util.Collections.*;

import org.junit.Before;
import org.junit.Test;

import static joptsimple.ExceptionMatchers.*;
import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class WExtensionWithArgumentTest extends AbstractOptionParserFixture {
    @Before
    public final void initializeParser() {
        parser.recognizeAlternativeLongOptions( true );
        parser.accepts( "silent" ).withOptionalArg();
    }

    @Test
    public void separateArgument() {
        OptionSet options = parser.parse( "-W", "silent", "4" );

        assertOptionNotDetected( options, "W" );
        assertOptionDetected( options, "silent" );
        assertEquals( singletonList( "4" ), options.valuesOf( "silent" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void togetherArgument() {
        OptionSet options = parser.parse( "-W", "silent=6" );

        assertOptionNotDetected( options, "W" );
        assertOptionDetected( options, "silent" );
        assertEquals( singletonList( "6" ), options.valuesOf( "silent" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void abbreviationWithTogetherArgument() {
        OptionSet options = parser.parse( "-W", "s=6" );

        assertOptionNotDetected( options, "W" );
        assertOptionDetected( options, "silent" );
        assertOptionNotDetected( options, "s" );
        assertEquals( singletonList( "6" ), options.valuesOf( "silent" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void illegalLongOption() {
        thrown.expect( UnrecognizedOptionException.class );
        thrown.expect( withOption( "foo" ) );

        parser.parse( "-W", "foo=bar" );
    }

    @Test
    public void noMoreArguments() {
        thrown.expect( OptionMissingRequiredArgumentException.class );
        thrown.expect( withOption( "W" ) );

        parser.parse( "-W" );
    }

    @Test
    public void typedTogetherArg() {
        OptionSpec<Integer> level = parser.accepts( "level" ).withRequiredArg().ofType( Integer.class );

        OptionSet options = parser.parse( "-W", "level=4" );

        assertOptionNotDetected( options, "W" );
        assertOptionDetected( options, "level" );
        assertEquals( singletonList( 4 ), options.valuesOf( level ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void typedSeparateArg() {
        OptionSpec<Integer> floor = parser.accepts( "floor" ).withRequiredArg().ofType( Integer.class );

        OptionSet options = parser.parse( "-W", "floor", "5" );

        assertOptionNotDetected( options, "W" );
        assertOptionDetected( options, "floor" );
        assertEquals( singletonList( 5 ), options.valuesOf( floor ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }
}
