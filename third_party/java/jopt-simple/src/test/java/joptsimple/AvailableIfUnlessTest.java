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

import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;

import static org.junit.Assert.assertEquals;

public class AvailableIfUnlessTest extends AbstractOptionParserFixture {
    @Before
    public void configureParser() {
        parser.mutuallyExclusive(
                parser.acceptsAll( asList( "ftp", "file-transfer" ) ),
                parser.accepts( "file" ),
                parser.accepts( "http" )
        );

        parser.acceptsAll( asList( "username", "userid" ) ).availableIf( "file-transfer" ).withRequiredArg();
        parser.acceptsAll( asList( "password", "pwd" ) ).availableIf( "ftp" ).withRequiredArg();
        parser.acceptsAll( asList( "directory", "dir" ) ).availableIf( "file" ).withRequiredArg();
        parser.accepts( "?" ).forHelp();
    }

    @Test
    public void rejectsEmptyMutualExclusiveness() {
        thrown.expect( NullPointerException.class );

        parser.mutuallyExclusive( (OptionSpecBuilder[]) null );
    }

    @Test
    public void rejectsConflictingCommandLineOptions1() {
        thrown.expect( UnavailableOptionException.class );

        parser.parse( "--ftp", "--file" );
    }

    @Test
    public void rejectsConflictingCommandLineOptions2() {
        thrown.expect( UnavailableOptionException.class );

        parser.parse( "--ftp", "--http" );
    }

    @Test
    public void rejectsIncompatibleOptions1() {
        thrown.expect( UnavailableOptionException.class );

        parser.parse( "--file", "--username", "joe" );
    }

    @Test
    public void rejectsIncompatibleOptions2() {
        thrown.expect( UnavailableOptionException.class );

        parser.parse( "--ftp", "--directory", "/tmp" );
    }


    @Test
    public void acceptsCommandLineWithConditionallyAllowedOptionsPresent1() {
        OptionSet options = parser.parse( "--ftp", "--userid", "joe", "--password=secret" );

        assertOptionDetected( options, "ftp" );
        assertOptionDetected( options, "username" );
        assertOptionDetected( options, "password" );
        assertEquals( singletonList( "joe" ), options.valuesOf( "username" ) );
        assertEquals( singletonList( "secret" ), options.valuesOf( "password" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void acceptsCommandLineWithConditionallyAllowedOptionsPresent2() {
        OptionSet options = parser.parse( "--file", "--dir", "/tmp");

        assertOptionDetected( options, "file" );
        assertOptionDetected( options, "dir" );
        assertEquals( singletonList( "/tmp" ), options.valuesOf( "dir" ) );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void rejectsCommandLineOnlyContainingForbiddenOptionSynonym() {
        thrown.expect( UnavailableOptionException.class );

        parser.parse( "--pwd", "secret" );
    }

    @Test
    public void rejectsOptionNotAlreadyConfigured() {
        thrown.expect( UnconfiguredOptionException.class );

        parser.accepts( "foo" ).availableIf( "bar" );
        parser.accepts( "foo" ).availableUnless( "bar" );
    }

    @Test
    public void presenceOfHelpOptionNegatesAllowedIfness() {
        OptionSet options = parser.parse( "--file", "--userid", "joe", "-?" );

        assertOptionDetected( options, "file" );
        assertOptionDetected( options, "?" );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }

    @Test
    public void presenceOfHelpOptionNegatesAllowedUnlessness() {
        OptionSet options = parser.parse( "--file", "--ftp",  "-?" );

        assertOptionDetected( options, "file" );
        assertOptionDetected( options, "file-transfer" );
        assertOptionDetected( options, "?" );
        assertEquals( emptyList(), options.nonOptionArguments() );
    }
}
