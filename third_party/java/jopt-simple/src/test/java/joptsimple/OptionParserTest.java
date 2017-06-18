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

import static java.lang.Boolean.*;
import static java.util.Arrays.*;
import static java.util.Collections.*;

import org.hamcrest.Description;
import org.hamcrest.TypeSafeMatcher;
import org.junit.Test;

import static joptsimple.ExceptionMatchers.*;
import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class OptionParserTest extends AbstractOptionParserFixture {
    @Test
    public void optionsAndNonOptionsInterspersed() {
        parser.accepts( "i" ).withOptionalArg();
        parser.accepts( "j" ).withOptionalArg();
        parser.accepts( "k" );

        OptionSet options =
            parser.parse( "-ibar", "-i", "junk", "xyz", "-jixnay", "foo", "-k", "blah", "--", "yermom" );

        assertOptionDetected( options, "i" );
        assertOptionDetected( options, "j" );
        assertOptionDetected( options, "k" );
        assertEquals( asList( "bar", "junk" ), options.valuesOf( "i" ) );
        assertEquals( singletonList( "ixnay" ), options.valuesOf( "j" ) );
        assertEquals( emptyList(), options.valuesOf( "k" ) );
        assertEquals( asList( "xyz", "foo", "blah", "yermom" ), options.nonOptionArguments() );
    }

    @Test
    public void shortOptionSpecifiedAsLongOptionWithoutArgument() {
        parser.accepts( "x" );

        OptionSet options = parser.parse( "--x" );
        assertOptionDetected( options, "x" );
        assertEquals( emptyList(), options.valuesOf( "x" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void longOptionLeadsWithSingleDash() {
        parser.accepts( "quiet" );
        parser.accepts( "queen" );

        OptionSet options = parser.parse( "-quiet" );
        assertOptionDetected( options, "quiet" );
        assertEquals( emptyList(), options.valuesOf( "quiet" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }
    
    @Test
    public void parsesLongOptionAsAbbreviatedShortOption() {
        parser.accepts( "queen" );

        parser.parse( "-que" );
    }
    
    @Test
    public void parsesLongOptionAsAbbreviatedLongOption() {
        parser.accepts( "queen" );

        parser.parse( "--que" );
    }

    @Test
    public void longOptionLeadsWithSingleDashAmbiguous() {
        parser.accepts( "quiet" );
        parser.accepts( "queen" );
        thrown.expect( UnrecognizedOptionException.class );
        thrown.expect( withOption( "q" ) );

        parser.parse( "-q" );
    }

    @Test
    public void longOptionLeadsWithSingleDashAmbiguousButShortsAreLegal() {
        parser.accepts( "quiet" );
        parser.accepts( "queen" );
        parser.accepts( "q" );
        parser.accepts( "u" );

        OptionSet options = parser.parse( "-qu" );
        assertOptionDetected( options, "q" );
        assertOptionDetected( options, "u" );
        assertOptionNotDetected( options, "quiet" );
        assertOptionNotDetected( options, "queen" );
        assertEquals( emptyList(), options.valuesOf( "q" ) );
        assertEquals( emptyList(), options.valuesOf( "u" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void longOptionLeadsWithSingleDashAmbiguousButAShortIsIllegal() {
        parser.accepts( "quiet" );
        parser.accepts( "queen" );
        parser.accepts( "q" );
        thrown.expect( UnrecognizedOptionException.class );
        thrown.expect( withOption( "u" ) );

        parser.parse( "-qu" );
    }

    @Test
    public void longOptionLeadsWithSingleDashAmbiguousButAShortAcceptsAnArgument() {
        parser.accepts( "quiet" );
        parser.accepts( "queen" );
        parser.accepts( "q" ).withOptionalArg();

        OptionSet options = parser.parse( "-qu" );
        assertOptionDetected( options, "q" );
        assertOptionNotDetected( options, "quiet" );
        assertOptionNotDetected( options, "queen" );
        assertEquals( singletonList( "u" ), options.valuesOf( "q" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void resetHappensAfterParsing() {
        parser.accepts( "i" ).withOptionalArg();
        parser.accepts( "j" ).withOptionalArg();
        parser.accepts( "k" );

        String[] args = { "-ibar", "-i", "junk", "xyz", "-jixnay", "foo", "-k", "blah", "--", "yermom" };

        OptionSet options = parser.parse( args );
        assertEquals( options, parser.parse( args ) );
    }

    @Test
    public void typedArguments() {
        parser.accepts( "a" ).withRequiredArg().ofType( Boolean.class );
        parser.accepts( "b" ).withOptionalArg().ofType( Integer.class );

        OptionSet options = parser.parse( "-a", "false", "-b", "3", "extra" );

        assertOptionDetected( options, "a" );
        assertOptionDetected( options, "b" );
        assertEquals( FALSE, options.valueOf( "a" ) );
        assertEquals( singletonList( FALSE ), options.valuesOf( "a" ) );
        assertEquals( Integer.valueOf( "3" ), options.valueOf( "b" ) );
        assertEquals( singletonList( Integer.valueOf( "3" ) ), options.valuesOf( "b" ) );
        assertEquals( singletonList( "extra" ), options.nonOptionArguments() );
    }

    @Test
    public void allowsMixingOfOptionsAndNonOptions() {
        parser.accepts( "i" ).withRequiredArg();
        parser.accepts( "j" ).withOptionalArg();
        parser.accepts( "k" );

        OptionSet options = parser.parse( "a", "b", "c", "-i", "boo", "d", "e", "-k", "f", "-j" );

        assertOptionDetected( options, "i" );
        assertEquals( singletonList( "boo" ), options.valuesOf( "i" ) );
        assertOptionDetected( options, "j" );
        assertEquals( emptyList(), options.valuesOf( "j" ) );
        assertOptionDetected( options, "k" );
        assertEquals( emptyList(), options.valuesOf( "k" ) );
        assertEquals( asList( "a", "b", "c", "d", "e", "f" ), options.nonOptionArguments() );
    }

    @Test
    public void disallowsMixingOfOptionsAndNonOptionsUnderPosixlyCorrect() {
        parser.accepts( "i" ).withRequiredArg();
        parser.accepts( "j" ).withOptionalArg();
        parser.accepts( "k" );
        parser.posixlyCorrect( true );

        OptionSet options = parser.parse( "a", "b", "c", "-i", "boo", "d", "e", "-k", "f", "-j" );

        assertOptionNotDetected( options, "i" );
        assertEquals( emptyList(), options.valuesOf( "i" ) );
        assertOptionNotDetected( options, "j" );
        assertEquals( emptyList(), options.valuesOf( "j" ) );
        assertOptionNotDetected( options, "k" );
        assertEquals( emptyList(), options.valuesOf( "j" ) );
        assertEquals( asList( "a", "b", "c", "-i", "boo", "d", "e", "-k", "f", "-j" ), options.nonOptionArguments() );
    }

    @Test
    public void doubleHyphenSignalsEndsOfOptions() {
        OptionSet options = new OptionParser( "ab:c::de:f::" ) {
            {
                accepts( "verbose" );
            }
        }.parse( "-a", "-b=foo", "-c=bar", "--", "-d", "-verbose", "-e", "baz", "-f", "biz" );

        assertOptionDetected( options, "a" );
        assertEquals( emptyList(), options.valuesOf( "a" ) );
        assertOptionDetected( options, "b" );
        assertEquals( singletonList( "foo" ), options.valuesOf( "b" ) );
        assertOptionDetected( options, "c" );
        assertEquals( singletonList( "bar" ), options.valuesOf( "c" ) );
        assertOptionNotDetected( options, "d" );
        assertOptionNotDetected( options, "verbose" );
        assertOptionNotDetected( options, "e" );
        assertOptionNotDetected( options, "f" );
        assertEquals( asList( "-d", "-verbose", "-e", "baz", "-f", "biz" ), options.nonOptionArguments() );
    }

    @Test
    public void allowsEmptyStringAsArgumentOfOption() {
        OptionSpec<String> optionI = parser.accepts( "i" ).withOptionalArg();

        OptionSet options = parser.parse( "-i", "" );

        assertOptionDetected( options, "i" );
        assertEquals( "", optionI.value( options ) );
    }

    @Test
    public void allowsWhitespaceyStringAsArgumentOfOption() {
        String whitespace = "     \t\t\n\n\f\f     \r\r   ";
        OptionSpec<String> optionJ = parser.accepts( "j" ).withRequiredArg();

        OptionSet options = parser.parse( "-j", whitespace );

        assertOptionDetected( options, "j" );
        assertEquals( whitespace, optionJ.value( options ) );
    }

    @Test
    public void allowsEmbeddedWhitespaceInArgumentOfOption() {
        String withEmbeddedWhitespace = "   look at me, I'm flaunting the rules!   ";
        OptionSpec<String> optionJ = parser.accepts( "j" ).withRequiredArg();

        OptionSet options = parser.parse( "-j", withEmbeddedWhitespace );

        assertOptionDetected( options, "j" );
        assertEquals( withEmbeddedWhitespace, optionJ.value( options ) );
    }
    
    @Test
    public void requiredOptionWithArgMissing() {
        parser.accepts( "t" ).withOptionalArg().required();
        thrown.expect( MissingRequiredOptionsException.class );
        thrown.expect( withOption( "t" ) );

        parser.parse();
    }

    @Test
    public void requiredOptionButHelpOptionPresent() {
        parser.accepts( "t" ).withOptionalArg().required();
        parser.accepts( "h" ).forHelp();

        parser.parse( "-h" );
    }

    @Test
    public void configurationPerformedLaterOverrideThosePerformedEarlierForTheSameOption() {
        parser.accepts( "t" ).withRequiredArg();
        parser.accepts( "t" ).withOptionalArg();

        parser.parse( "-t" );
    }

    @Test
    public void requiredOptionWithSynonymsMissing() {
        parser.acceptsAll( asList( "h", "help", "?" ) );
        parser.acceptsAll( asList( "f", "ff", "csv-file-name" ) ).withRequiredArg().required();

        thrown.expect( MissingRequiredOptionsException.class );
        thrown.expectMessage( new TypeSafeMatcher<String>() {
            @Override
            protected boolean matchesSafely( String item ) {
                return "Missing required option(s) [f/csv-file-name/ff]".equals( item );
            }

            public void describeTo(Description description) {
                // purposely doing nothing here
            }
        } );

        parser.parse();
    }

    @Test
    public void abbreviationsCanBeDisallowed() {
        OptionParser parser = new OptionParser(false);
        parser.accepts( "abbreviatable" );

        thrown.expect( UnrecognizedOptionException.class );

        parser.parse( "--abb" );
    }
}
