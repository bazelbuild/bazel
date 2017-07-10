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

import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;
import static org.junit.Assert.assertEquals;

public class RequiredUnlessTest extends AbstractOptionParserFixture {
    @Before
    public void configureParser() {
        OptionSpec<Void> anonymous = parser.accepts( "anonymous" );
        parser.acceptsAll( asList( "username", "userid" ) ).requiredUnless( "anonymous" ).withRequiredArg();
        parser.acceptsAll( asList( "password", "pwd" ) ).requiredUnless( anonymous ).withRequiredArg();
        parser.accepts( "?" ).forHelp();
    }

    @Test
    public void rejectsCommandLineMissingConditionallyRequiredOption() {
        thrown.expect( MissingRequiredOptionsException.class );

        parser.parse( "" );
    }

    @Test
    public void rejectsCommandLineWithNotAllConditionallyRequiredOptionsPresent() {
        thrown.expect( MissingRequiredOptionsException.class );

        parser.parse( "--username", "joe" );
    }

    @Test
    public void acceptsCommandLineWithConditionallyRequiredOptionsPresent() {
        OptionSet options = parser.parse( "--userid", "joe", "--password=secret" );

        assertOptionNotDetected( options, "anonymous" );
        assertOptionDetected( options, "username" );
        assertOptionDetected( options, "password" );
        assertEquals( singletonList( "joe" ), options.valuesOf( "username" ) );
        assertEquals( singletonList( "secret" ), options.valuesOf( "password" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void acceptsOptionWithPrerequisite() {
        OptionSet options = parser.parse( "--anonymous" );

        assertOptionDetected( options, "anonymous" );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void acceptsOptionWithPrerequisiteAsNormal() {
        OptionSet options = parser.parse( "--anonymous", "--pwd", "secret" );

        assertOptionDetected( options, "anonymous" );
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
    public void presenceOfHelpOptionNegatesRequiredUnlessness() {
        OptionSet options = parser.parse( "-?" );

        assertOptionDetected( options, "?" );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }
}
