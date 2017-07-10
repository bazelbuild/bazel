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

package joptsimple.util;

import joptsimple.ValueConversionException;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static joptsimple.util.RegexMatcher.*;
import static org.junit.Assert.*;
import static org.junit.rules.ExpectedException.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class RegexMatcherTest {
    @Rule public final ExpectedException thrown = none();

    private RegexMatcher abc;

    @Before
    public void setUp() {
        abc = new RegexMatcher( "abc", 0 );
    }

    @Test
    public void shouldAttemptToMatchValueAgainstARegex() {
        assertEquals( "abc", abc.convert( "abc" ) );
    }

    @Test( expected = ValueConversionException.class )
    public void rejectsValueThatDoesNotMatchRegex() {
        abc.convert( "abcd" );
    }

    @Test
    public void raisesExceptionContainingValueAndPattern() {
        thrown.expect( ValueConversionException.class );
        thrown.expectMessage( "\\d+" );
        thrown.expectMessage( "asdf" );

        new RegexMatcher( "\\d+", 0 ).convert( "asdf" );
    }

    @Test
    public void shouldOfferConvenienceMethodForCreatingMatcherWithNoFlags() {
        assertEquals( "sourceforge.net", regex( "\\w+\\.\\w+" ).convert( "sourceforge.net" ) );
    }

    @Test
    public void shouldAnswerCorrectValueType() {
        assertEquals( String.class, abc.valueType() );
    }

    @Test
    public void shouldGiveCorrectValuePattern() {
        assertEquals( "abc", abc.valuePattern() );
    }
}
