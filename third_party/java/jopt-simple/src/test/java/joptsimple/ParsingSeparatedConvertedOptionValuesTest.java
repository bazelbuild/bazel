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

import static java.util.Arrays.*;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class ParsingSeparatedConvertedOptionValuesTest extends AbstractOptionParserFixture {
    @Test
    public void parsesSeparatedValuesAsSeparateArgument() {
        assertCorrectParse( "classpath", ':', "--classpath", "/usr:/opt:/var" );
    }

    @Test
    public void parsesSeparatedValuesWhenArgumentAbuttedWithEquals() {
        assertCorrectParse( "classpath", ':', "--classpath=/usr:/opt:/var" );
    }

    @Test
    public void parsesEqualsSeparatedValuesWhenArgumentAbuttedWithEquals() {
        assertCorrectParse( "classpath", '=', "--classpath=/usr=/opt=/var" );
    }

    @Test
    public void parsesSeparatedValuesAbutting() {
        assertCorrectParse( "c", ':', "-c/usr:/opt:/var" );
    }

    private void assertCorrectParse( String option, char separator, String... args ) {
        parser.accepts( option ).withRequiredArg().withValuesSeparatedBy( separator )
            .withValuesConvertedBy( new ValueConverter<File>() {
                public File convert( String value ) {
                    return new File( value );
                }

                public Class<File> valueType() {
                    return File.class;
                }

                public String valuePattern() {
                    return null;
                }
            } );

        OptionSet options = parser.parse( args );

        assertEquals(
            asList( new File( "/usr" ), new File( "/opt" ), new File( "/var" ) ),
            options.valuesOf( option ) );
    }
}
