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

import org.junit.Before;
import org.junit.Test;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;

public class RequiredUnlessAnyTest extends AbstractOptionParserFixture {
    @Before
    public void configureParser() {
        parser.accepts( "a" );
        parser.accepts( "b" );
        OptionSpec<Void> c = parser.accepts( "c" );
        OptionSpec<Void> d = parser.accepts( "d" );
        parser.accepts( "e" );
        parser.accepts( "n" ).requiredUnless( "a", "b" ).requiredUnless( c, d );
    }

    @Test
    public void commandLineMissingConditionallyRequiredOption() {
        OptionSet options = parser.parse( "-a" );
        assertOptionDetected( options, "a" );
        assertOptionNotDetected( options, "n" );
    }

    @Test
    public void commandLineWithNotAllConditionallyRequiredOptionsPresent() {
        OptionSet options = parser.parse( "-a", "-b", "-c", "-d" );
        assertOptionDetected( options, "a" );
        assertOptionDetected( options, "b" );
        assertOptionDetected( options, "c" );
        assertOptionDetected( options, "d" );
        assertOptionNotDetected( options, "n" );
    }

    @Test
    public void acceptsCommandLineWithConditionallyRequiredOptionsPresent() {
        OptionSet options = parser.parse( "-n" );

        assertOptionDetected( options, "n" );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void acceptsOptionWithPrerequisiteAsNormalIfPrerequisiteNotInPlay() {
        OptionSet options = parser.parse( "-a", "-n" );

        assertOptionDetected( options, "a" );
        assertOptionDetected( options, "n" );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }
}
