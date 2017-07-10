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

import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class RequiredIfTest extends AbstractOptionParserFixture {
    @Before
    public void configureParser() {
        OptionSpec<Void> ftp = parser.acceptsAll( asList( "ftp", "file-transfer" ) );
        parser.acceptsAll( asList( "username", "userid" ) ).requiredIf( "file-transfer" ).withRequiredArg();
        parser.acceptsAll( asList( "password", "pwd" ) ).requiredIf( ftp ).withRequiredArg();
        parser.accepts( "?" ).forHelp();
    }

    @Test
    public void rejectsCommandLineMissingConditionallyRequiredOption() {
        thrown.expect( MissingRequiredOptionsException.class );

        parser.parse( "--ftp" );
    }

    @Test
    public void rejectsCommandLineMissingConditionallyRequiredOptionSynonym() {
        thrown.expect( MissingRequiredOptionsException.class );

        parser.parse( "--file-transfer" );
    }

    @Test
    public void rejectsCommandLineWithNotAllConditionallyRequiredOptionsPresent() {
        thrown.expect( MissingRequiredOptionsException.class );

        parser.parse( "--ftp", "--username", "joe" );
    }

    @Test
    public void acceptsCommandLineWithConditionallyRequiredOptionsPresent() {
        OptionSet options = parser.parse( "--ftp", "--userid", "joe", "--password=secret" );

        assertOptionDetected( options, "ftp" );
        assertOptionDetected( options, "username" );
        assertOptionDetected( options, "password" );
        assertEquals( singletonList( "joe" ), options.valuesOf( "username" ) );
        assertEquals( singletonList( "secret" ), options.valuesOf( "password" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void acceptsOptionWithPrerequisiteAsNormalIfPrerequisiteNotInPlay() {
        OptionSet options = parser.parse( "--pwd", "secret" );

        assertOptionDetected( options, "pwd" );
        assertEquals( singletonList( "secret" ), options.valuesOf( "pwd" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void rejectsOptionNotAlreadyConfigured() {
        thrown.expect( UnconfiguredOptionException.class );

        parser.accepts( "foo" ).requiredIf( "bar" );
    }

    @Test
    public void rejectsOptionSpecNotAlreadyConfigured() {
        thrown.expect( UnconfiguredOptionException.class );

        parser.accepts( "foo" ).requiredIf( "bar" );
    }

    @Test
    public void presenceOfHelpOptionNegatesRequiredIfness() {
        OptionSet options = parser.parse( "--ftp", "-?" );

        assertOptionDetected( options, "ftp" );
        assertOptionDetected( options, "?" );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }
}
