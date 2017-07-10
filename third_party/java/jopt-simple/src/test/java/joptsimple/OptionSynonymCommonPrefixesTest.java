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

import org.junit.Before;
import org.junit.Test;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class OptionSynonymCommonPrefixesTest extends AbstractOptionParserFixture {
    @Before
    public final void initializeParser() {
        parser.acceptsAll( asList( "v", "verbose" ) );
    }

    @Test
    public void parsingFirstPrefix() {
        assertDetections( "-v" );
    }

    @Test
    public void parsingSecondPrefix() {
        assertDetections( "-ve" );
    }

    @Test
    public void parsingThirdPrefix() {
        assertDetections( "-ver" );
    }

    @Test
    public void parsingFourthPrefix() {
        assertDetections( "-verb" );
    }

    @Test
    public void parsingFifthPrefix() {
        assertDetections( "-verbo" );
    }

    @Test
    public void parsingSixthPrefix() {
        assertDetections( "-verbos" );
    }

    @Test
    public void parsingSeventhPrefix() {
        assertDetections( "-verbose" );
    }

    private void assertDetections( String option ) {
        OptionSet options = parser.parse( option );

        assertOptionDetected( options, "v" );
        assertOptionNotDetected( options, "ve" );
        assertOptionNotDetected( options, "ver" );
        assertOptionNotDetected( options, "verb" );
        assertOptionNotDetected( options, "verbo" );
        assertOptionNotDetected( options, "verbos" );
        assertOptionDetected( options, "verbose" );
    }
}
