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

import static joptsimple.ExceptionMatchers.*;
import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class ShortOptionsOptionalArgumentTest extends AbstractOptionParserFixture {
    @Before
    public final void initializeParser() {
        parser.accepts( "f" ).withOptionalArg();
        parser.accepts( "g" ).withOptionalArg();
        parser.accepts( "bar" ).withOptionalArg();
    }

    @Test
    public void optionWithOptionalArgumentNotPresent() {
        OptionSet options = parser.parse( "-f" );

        assertOptionDetected( options, "f" );
        assertEquals( emptyList(), options.valuesOf( "f" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void optionWithOptionalArgumentPresent() {
        OptionSet options = parser.parse( "-f", "bar" );

        assertOptionDetected( options, "f" );
        assertEquals( singletonList( "bar" ), options.valuesOf( "f" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void optionWithOptionalArgumentThatLooksLikeAnInvalidOption() {
        thrown.expect( UnrecognizedOptionException.class );
        thrown.expect( withOption( "biz" ) );

        parser.parse( "-f", "--biz" );
    }

    @Test
    public void optionWithOptionalArgumentThatLooksLikeAValidOption() {
        OptionSet options = parser.parse( "-f", "--bar" );

        assertOptionDetected( options, "f" );
        assertOptionDetected( options, "bar" );
        assertEquals( emptyList(), options.valuesOf( "f" ) );
        assertEquals( emptyList(), options.valuesOf( "bar" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void optionWithOptionalArgumentFollowedByLegalOption() {
        OptionSet options = parser.parse( "-f", "-g" );

        assertOptionDetected( options, "f" );
        assertOptionDetected( options, "g" );
        assertEquals( emptyList(), options.valuesOf( "f" ) );
        assertEquals( emptyList(), options.valuesOf( "g" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void multipleOfSameOptionSomeWithArgsAndSomeWithout() {
        OptionSet options = parser.parse( "-f", "-f", "foo", "-f", "-f", "bar", "-f" );

        assertEquals( asList( "foo", "bar" ), options.valuesOf( "f" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }
}
