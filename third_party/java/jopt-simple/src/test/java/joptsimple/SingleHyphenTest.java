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

import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class SingleHyphenTest extends AbstractOptionParserFixture {
    @Before
    public final void initializeParser() {
        parser.accepts( "o" ).withOptionalArg();
    }

    @Test
    public void singleHyphen() {
        OptionSet options = parser.parse( "-" );

        assertEquals( singletonList( "-" ), options.nonOptionArguments() );
    }

    @Test
    public void asArgumentOfOption() {
        OptionSet options = parser.parse( "-o", "-" );

        assertOptionDetected( options, "o" );
        assertEquals( singletonList( "-" ), options.valuesOf( "o" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void asArgumentOfOptionInSameToken() {
        OptionSet options = parser.parse( "-o-" );

        assertOptionDetected( options, "o" );
        assertEquals( singletonList( "-" ), options.valuesOf( "o" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }
}
