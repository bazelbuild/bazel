/*
 The MIT License

 Copyright (c) 2004-2014 Paul R. Holser, Jr.

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

import org.junit.Before;
import org.junit.Test;

import static java.util.Collections.emptyList;

import static org.junit.Assert.assertEquals;

public class AvailableIfAnyTest extends AbstractOptionParserFixture {
    @Before
    public void configureParser() {
        OptionSpec<Void> a = parser.accepts( "a" );
        parser.accepts( "b" );
        OptionSpec<Void> c = parser.accepts( "c" );
        parser.accepts( "d" );
        parser.accepts( "n" ).availableIf( a, c );
    }

    @Test
    public void rejectsCommandLineExistingForbiddenOption() {
        thrown.expect( UnavailableOptionException.class );

        parser.parse( "-b", "-n" );
    }

    @Test
    public void rejectsCommandLineExistingOtherForbiddenOption() {
        thrown.expect( UnavailableOptionException.class );

        parser.parse( "-d", "-n" );
    }

    @Test
    public void rejectsCommandLineOnlyForbiddenOption() {
        thrown.expect( UnavailableOptionException.class );

        parser.parse( "-n" );
    }

    @Test
    public void rejectsCommandLineWithNotAllConditionallyRequiredOptionsPresent() {
        thrown.expect( UnavailableOptionException.class );

        parser.parse( "-b", "-d", "-n" );
    }

    @Test
    public void acceptsCommandLineWithAllowedOptionsPresent() {
        OptionSet options = parser.parse( "-a", "-c", "-n" );
        
        assertOptionDetected( options, "a" );
        assertOptionDetected( options, "c" );
        assertOptionDetected( options, "n" );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }
}
