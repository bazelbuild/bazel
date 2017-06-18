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

import java.io.File;
import java.util.List;

import static java.util.Arrays.*;
import static java.util.Collections.*;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class OptionSynonymOptionalArgumentTest extends AbstractOptionParserFixture {
    private String optionArgument;

    @Before
    public final void initializeParser() {
        parser.acceptsAll( asList( "d", "output-dir", "folder" ), "file" ).withRequiredArg().ofType( File.class );
        optionArgument = "tmp";
    }

    @Test
    public void hasAllSynonymsWhenFirstSynonymParsed() {
        assertDetections( new String[] { "-d", optionArgument }, singletonList( new File( optionArgument ) ) );
    }

    @Test
    public void hasAllSynonymsWhenSecondSynonymParsed() {
        assertDetections( new String[] { "--ou", optionArgument }, singletonList( new File( optionArgument ) ) );
    }

    @Test
    public void hasAllSynonymsWhenThirdSynonymParsed() {
        assertDetections( new String[] { "--fold", optionArgument }, singletonList( new File( optionArgument ) ) );
    }

    @Test
    public void reportsSameListOfArgumentsForEverySynonymOption() {
        assertDetections(
            new String[] { "-d", "opt", "--output-dir", "tmp", "--folder", "usr" },
            asList( new File( "opt" ), new File( "tmp" ), new File( "usr" ) ) );
    }

    private void assertDetections( String[] args, List<?> optionArguments ) {
        OptionSet options = parser.parse( args );
        assertEquals( optionArguments, options.valuesOf( "d" ) );
        assertEquals( optionArguments, options.valuesOf( "output-dir" ) );
        assertEquals( optionArguments, options.valuesOf( "folder" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }
}
