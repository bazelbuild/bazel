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

import java.util.Collection;

import static java.util.Arrays.*;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
@RunWith( Parameterized.class )
public class PosixlyCorrectOptionParserTest {
    private final OptionParser parser;

    public PosixlyCorrectOptionParserTest( OptionParser parser ) {
        this.parser = parser;
    }

    @Parameterized.Parameters
    public static Collection<?> parsers() {
        return asList( new Object[] {
            new OptionParser() { {
                posixlyCorrect( true );
                accepts( "i" ).withRequiredArg();
                accepts( "j" ).withOptionalArg();
                accepts( "k" );
            } } },
            new Object[] { new OptionParser( "+i:j::k" ) } );
    }

    @Test
    public void parseWithPosixlyCorrect() {
        OptionSet options =
            parser.parse( "-ibar", "-i", "junk", "xyz", "-jixnay", "foo", "-k", "blah", "--", "yermom" );

        assertTrue( "i?", options.has( "i" ) );
        assertFalse( "j?", options.has( "j" ) );
        assertFalse( "k?", options.has( "k" ) );
        assertEquals( "args of i?", asList( "bar", "junk" ), options.valuesOf( "i" ) );
        assertEquals( "non-option args?",
            asList( "xyz", "-jixnay", "foo", "-k", "blah", "--", "yermom" ),
            options.nonOptionArguments() );
    }
}
